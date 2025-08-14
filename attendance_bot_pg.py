# attendance_bot_pg.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import csv
import io
import logging
from datetime import datetime, timezone
from typing import Optional, Sequence

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    ChatMemberHandler, MessageHandler, ContextTypes, filters
)
from telegram.constants import ChatMemberStatus, ParseMode

from sqlalchemy import (
    String,
    Integer,
    BigInteger,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    select,
    func,
)
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# -------------------- Logging --------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=getattr(logging, LOG_LEVEL, logging.INFO),
)
logger = logging.getLogger("attendance-bot")

# -------------------- Config --------------------
TOKEN = os.environ.get("BOT_TOKEN", "")
DB_USER = os.environ.get("POSTGRES_USER", "")
DB_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "")
DB_HOST = os.environ.get("POSTGRES_HOST", "localhost")
DB_NAME = os.environ.get("POSTGRES_DB", "postgres")
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}",
)
# اگر خالی باشد، برای راحتی توسعه همه اجازه دارند (قبل از دیپلوی مقدار بدهید)
ADMIN_IDS: set[int] = set(
    int(x) for x in os.environ.get("ADMIN_IDS", "").replace(" ", "").split(",") if x
)

if not TOKEN:
    raise RuntimeError("BOT_TOKEN env is required")

# -------------------- ORM Models --------------------
class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "tg_users"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    username: Mapped[Optional[str]] = mapped_column(String(255))
    first_name: Mapped[Optional[str]] = mapped_column(String(255))
    last_name: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

class Channel(Base):
    __tablename__ = "channels"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)  # -100...
    title: Mapped[Optional[str]] = mapped_column(String(255))
    type: Mapped[str] = mapped_column(String(32), default="channel")  # channel/supergroup
    bot_status: Mapped[Optional[str]] = mapped_column(String(64))  # administrator/member/left/...
    added_by_user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("tg_users.id"))
    added_by_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    roles: Mapped[list["UserChannelRole"]] = relationship(back_populates="channel")

class UserChannelRole(Base):
    __tablename__ = "user_channel_roles"
    __table_args__ = (UniqueConstraint("user_id", "channel_id", name="uq_user_channel_role"),)
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("tg_users.id", ondelete="CASCADE"))
    channel_id: Mapped[int] = mapped_column(ForeignKey("channels.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(32), default="owner")  # owner/manager
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    channel: Mapped[Channel] = relationship(back_populates="roles")

class Post(Base):
    __tablename__ = "posts"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    channel_message_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

class Attendance(Base):
    __tablename__ = "attendance"
    __table_args__ = (UniqueConstraint("post_id", "user_id", name="uq_attendance_post_user"),)
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    post_id: Mapped[int] = mapped_column(ForeignKey("posts.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("tg_users.id", ondelete="CASCADE"), index=True)
    clicked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

# -------------------- DB Engine & Session --------------------
engine = create_async_engine(DATABASE_URL, echo=False, future=True)
Session = async_sessionmaker(engine, expire_on_commit=False)

# -------------------- Helpers --------------------
def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS if ADMIN_IDS else True

ROWS_PER_PAGE = 30

def _format_name(username: str | None, first_name: str | None, last_name: str | None, user_id: int) -> tuple[str, str]:
    user_col = f"@{username}" if username else str(user_id)
    name_col = " ".join([x for x in [first_name, last_name] if x]) or "-"
    return user_col, name_col

def _render_stats_table(rows, page: int, per_page: int = ROWS_PER_PAGE) -> tuple[str, int, int]:
    total = len(rows)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    end = min(start + per_page, total)
    visible = rows[start:end]

    header = f"{'#':>3}  {'User':<20}  {'Name':<24}  {'Time (UTC)':<19}"
    sep = "-" * len(header)
    lines = [header, sep]
    for idx, r in enumerate(visible, start=start + 1):
        # r = (user_id, username, first_name, last_name, clicked_at)
        uid, uname, fn, ln, ts = r
        user_col, name_col = _format_name(uname, fn, ln, uid)
        ts_str = ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"{idx:>3}  {user_col:<20.20}  {name_col:<24.24}  {ts_str:<19}")
    body = "\n".join(lines) if visible else "— هنوز کسی کلیک نکرده —"
    html = f"<pre>{body}</pre>"
    return html, page, total_pages

def _stats_kb(post_id: int, page: int, total_pages: int) -> InlineKeyboardMarkup | None:
    nav = []
    if page > 1:
        nav.append(InlineKeyboardButton("« قبلی", callback_data=f"statspg:{post_id}:{page-1}"))
    if page < total_pages:
        nav.append(InlineKeyboardButton("بعدی »", callback_data=f"statspg:{post_id}:{page+1}"))
    return InlineKeyboardMarkup([nav]) if nav else None


def kb(count: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(text=f"حاضر ✅ ({count})", callback_data="present")]]
    )

async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_or_create_user(session, tg_user) -> User:
    q = await session.execute(select(User).where(User.user_id == tg_user.id))
    user = q.scalar_one_or_none()
    if user is None:
        user = User(
            user_id=tg_user.id,
            username=tg_user.username,
            first_name=tg_user.first_name,
            last_name=tg_user.last_name,
        )
        session.add(user)
        await session.flush()
    else:
        changed = False
        if user.username != tg_user.username:
            user.username = tg_user.username; changed = True
        if user.first_name != tg_user.first_name:
            user.first_name = tg_user.first_name; changed = True
        if user.last_name != tg_user.last_name:
            user.last_name = tg_user.last_name; changed = True
        if changed:
            await session.flush()
    return user

async def get_or_create_post(session, channel_message_id: int) -> Post:
    q = await session.execute(select(Post).where(Post.channel_message_id == channel_message_id))
    post = q.scalar_one_or_none()
    if post is None:
        post = Post(channel_message_id=channel_message_id)
        session.add(post)
        await session.flush()
    return post

async def attendance_count(session, post_id: int) -> int:
    q = await session.execute(select(func.count(Attendance.id)).where(Attendance.post_id == post_id))
    return int(q.scalar_one())

# -------------------- Channel list UI --------------------
PAGE_SIZE = 8

def channels_keyboard(rows: Sequence[Channel], page: int, total_pages: int) -> InlineKeyboardMarkup:
    btn_rows = []
    for ch in rows:
        title = ch.title or str(ch.chat_id)
        btn_rows.append([InlineKeyboardButton(title, callback_data=f"pickch:{ch.chat_id}")])
    nav = []
    if page > 1:
        nav.append(InlineKeyboardButton("« قبلی", callback_data=f"chpage:{page-1}"))
    if page < total_pages:
        nav.append(InlineKeyboardButton("بعدی »", callback_data=f"chpage:{page+1}"))
    if nav:
        btn_rows.append(nav)
    return InlineKeyboardMarkup(btn_rows)

async def list_user_channels(session, user: User, page: int = 1):
    # کانال‌هایی که کاربر نقشی دارد و بات هم در آن عضو/ادمین است
    base = (
        select(Channel)
        .join(UserChannelRole, UserChannelRole.channel_id == Channel.id)
        .where(UserChannelRole.user_id == user.id)
        .where(Channel.bot_status.in_(["administrator", "member"]))
    )
    total = (await session.execute(select(func.count()).select_from(base.subquery()))).scalar_one()
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(1, min(page, total_pages))
    rows = (
        await session.execute(
            base.order_by(Channel.title.nulls_last()).offset((page - 1) * PAGE_SIZE).limit(PAGE_SIZE)
        )
    ).scalars().all()
    return rows, page, total_pages

# -------------------- Handlers --------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "سلام! من بات حضور کانالم.\n"
        "• بات را ادمین کانالت کن تا ثبت شود.\n"
        "• /post برای ارسال پست حضور (ابتدا از لیست کانال‌ها انتخاب کن)\n"
        "• /export برای خروجی CSV\n"
        "• /stats برای شمارش آخرین پست"
    )

async def on_my_chat_member(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """وقتی بات به کانالی اضافه/ادمین می‌شود یا وضعیتش عوض می‌شود."""
    m = update.my_chat_member
    if not m or m.chat.type not in ("channel", "supergroup"):
        return
    actor = m.from_user
    chat = m.chat
    new_status = getattr(m.new_chat_member.status, "value", str(m.new_chat_member.status))
    old_status = getattr(m.old_chat_member.status, "value", str(m.old_chat_member.status))

    await init_db()
    async with Session() as session:
        user = await get_or_create_user(session, actor)

        # Upsert Channel
        q = await session.execute(select(Channel).where(Channel.chat_id == chat.id))
        ch = q.scalar_one_or_none()
        if ch is None:
            ch = Channel(
                chat_id=chat.id,
                title=chat.title,
                type=chat.type,
                bot_status=new_status,
                added_by_user_id=user.id,
            )
            session.add(ch)
            await session.flush()
        else:
            ch.title = chat.title or ch.title
            ch.type = chat.type
            ch.bot_status = new_status
            ch.last_seen_at = datetime.now(timezone.utc)
            if ch.added_by_user_id is None:
                ch.added_by_user_id = user.id

        # اگر بات ادمین شد، نقش owner برای actor ثبت شود
        if new_status == ChatMemberStatus.ADMINISTRATOR.value:
            q = await session.execute(
                select(UserChannelRole).where(
                    UserChannelRole.user_id == user.id, UserChannelRole.channel_id == ch.id
                )
            )
            role = q.scalar_one_or_none()
            if role is None:
                session.add(UserChannelRole(user_id=user.id, channel_id=ch.id, role="owner"))

        await session.commit()

    logger.info(
        "my_chat_member: chat_id=%s title=%r old=%s new=%s actor=%s",
        chat.id,
        chat.title,
        old_status,
        new_status,
        actor.id,
    )

async def cmd_post(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """اگر آرگومان ندهی، از لیست کانال‌های خودت انتخاب می‌کنی و بعد پست ارسال می‌شود."""
    u = update.effective_user
    if not is_admin(u.id):
        return await update.message.reply_text("⛔️ دسترسی نداری.")
    await init_db()
    async with Session() as session:
        user = await get_or_create_user(session, u)
        rows, page, total_pages = await list_user_channels(session, user, page=1)
    if not rows:
        return await update.message.reply_text(
            "هیچ کانالی برای شما ثبت نشده یا بات در آن عضو/ادمین نیست. "
            "بات را به کانالت اَد و ادمین کن، سپس دوباره /post بزن."
        )
    await update.message.reply_text(
        "یکی از کانال‌ها را انتخاب کن تا پست حضور ارسال شود:",
        reply_markup=channels_keyboard(rows, page, total_pages),
    )

async def on_pick_channel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """صفحه‌بندی/انتخاب کانال جهت ارسال پست حضور."""
    q = update.callback_query
    data = q.data
    await init_db()
    if data.startswith("chpage:"):
        page = int(data.split(":")[1])
        async with Session() as session:
            user = await get_or_create_user(session, q.from_user)
            rows, page, total_pages = await list_user_channels(session, user, page=page)
        await q.edit_message_reply_markup(reply_markup=channels_keyboard(rows, page, total_pages))
        return await q.answer()
    if data.startswith("pickch:"):
        chat_id = int(data.split(":")[1])
        # تلاش برای ارسال پیام؛ اگر دسترسی نباشد، استثنا می‌گیریم
        try:
            msg = await context.bot.send_message(
                chat_id=chat_id,
                text="اگر این پست را دیدی، روی دکمه بزن 👇",
                reply_markup=kb(0),
                disable_web_page_preview=True,
            )
        except Exception as e:
            logger.warning("send_message failed to %s: %s", chat_id, e)
            return await q.answer("دسترسی به کانال ندارم یا ادمین نیستم.", show_alert=True)

        # در DB ثبت پست
        async with Session() as session:
            await get_or_create_post(session, msg.message_id)
            await session.commit()

        await q.answer("ارسال شد ✅")
        await q.edit_message_text(
            f"ارسال شد به کانال `{chat_id}` (پیام #{msg.message_id})", parse_mode="Markdown"
        )

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """ثبت کلیک دکمهٔ حضور زیر پست کانال + جلوگیری از دوباره‌زنی + آپدیت شمارنده."""
    if not update.callback_query:
        return
    q = update.callback_query
    if not q.message or q.message.chat.type != "channel":
        return await q.answer("فقط روی پست کانال معتبر است.", show_alert=True)

    await init_db()
    async with Session() as session:
        post = await get_or_create_post(session, q.message.message_id)
        user = await get_or_create_user(session, q.from_user)

        # آیا قبلاً برای این پست حاضر زده؟
        already = (await session.execute(
            select(func.count(Attendance.id)).where(
                Attendance.post_id == post.id,
                Attendance.user_id == user.id
            )
        )).scalar_one() > 0

        if already:
            # شمارنده فعلی را بخوان (تغییری ندارد)
            n = await attendance_count(session, post.id)
            try:
                await q.message.edit_reply_markup(reply_markup=kb(n))
            except Exception:
                pass
            return await q.answer("یه بار حاضری زدی 👀", show_alert=True)

        # اولین بار است: ثبت حضور
        session.add(Attendance(post_id=post.id, user_id=user.id))
        await session.commit()

        n = await attendance_count(session, post.id)

    try:
        await q.message.edit_reply_markup(reply_markup=kb(n))
    except Exception:
        pass
    await q.answer("ثبت شد ✅", show_alert=False)


async def cmd_export(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """خروجی CSV از آخرین پست یا یک message_id مشخص (ادمین داخلی)."""
    u = update.effective_user
    if not is_admin(u.id):
        return await update.message.reply_text("⛔️ فقط ادمین می‌تواند خروجی بگیرد.")
    await init_db()
    # optional channel_message_id
    target_msg_id: Optional[int] = None
    if context.args:
        try:
            target_msg_id = int(context.args[0])
        except ValueError:
            return await update.message.reply_text("فرمت: /export [channel_message_id]")

    async with Session() as session:
        post: Optional[Post]
        if target_msg_id is not None:
            post = (
                await session.execute(select(Post).where(Post.channel_message_id == target_msg_id))
            ).scalar_one_or_none()
            if not post:
                return await update.message.reply_text("پستی با این شناسه در DB نیست.")
        else:
            post = (await session.execute(select(Post).order_by(Post.id.desc()).limit(1))).scalar_one_or_none()
            if not post:
                return await update.message.reply_text("هنوز پستی ثبت نشده.")
        rows = (
            await session.execute(
                select(User.user_id, User.username, User.first_name, User.last_name, Attendance.clicked_at)
                .join(Attendance, Attendance.user_id == User.id)
                .where(Attendance.post_id == post.id)
                .order_by(Attendance.clicked_at)
            )
        ).all()

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["user_id", "username", "first_name", "last_name", "clicked_at"])
    for r in rows:
        clicked_iso = r[4].astimezone(timezone.utc).isoformat() if isinstance(r[4], datetime) else str(r[4])
        w.writerow([r[0], r[1] or "", r[2] or "", r[3] or "", clicked_iso])
    buf.seek(0)

    caption = f"حاضرین پست #{post.channel_message_id} — تعداد: {len(rows)}"
    await update.message.reply_document(
        document=("attendance.csv", buf.getvalue().encode("utf-8")), caption=caption
    )

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    if not is_admin(u.id):
        return await update.message.reply_text("⛔️ فقط ادمین.")
    await init_db()

    # آرگ اختیاری: channel_message_id
    target_msg_id: Optional[int] = None
    if context.args:
        try:
            target_msg_id = int(context.args[0])
        except ValueError:
            return await update.message.reply_text("فرمت: /stats [channel_message_id]")

    async with Session() as session:
        if target_msg_id is None:
            post = (await session.execute(select(Post).order_by(Post.id.desc()).limit(1))).scalar_one_or_none()
            if not post:
                return await update.message.reply_text("هنوز پستی ثبت نشده.")
        else:
            post = (await session.execute(select(Post).where(Post.channel_message_id == target_msg_id))).scalar_one_or_none()
            if not post:
                return await update.message.reply_text("پستی با این شناسه در DB نیست.")

        rows = (await session.execute(
            select(User.user_id, User.username, User.first_name, User.last_name, Attendance.clicked_at)
            .join(Attendance, Attendance.user_id == User.id)
            .where(Attendance.post_id == post.id)
            .order_by(Attendance.clicked_at)
        )).all()

    table_html, page, total_pages = _render_stats_table(rows, page=1, per_page=ROWS_PER_PAGE)
    header = f"آمار پست #{post.channel_message_id} — کل: {len(rows)}\n"
    await update.message.reply_text(
        header + table_html,
        parse_mode=ParseMode.HTML,
        reply_markup=_stats_kb(post.id, page, total_pages)
    )


async def on_channel_post(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # به‌خاطر فیلترِ ChatType.CHANNEL این پیام حتماً پست کاناله
    msg = update.effective_message
    if not msg:
        return
    logger.info(
        "channel_post -> chat_title=%r chat_id=%s message_id=%s text=%r",
        msg.chat.title, msg.chat.id, msg.message_id, (msg.text or "")[:80]
    )

async def on_stats_page(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    _, post_id_str, page_str = q.data.split(":")
    post_id = int(post_id_str); page = int(page_str)

    await init_db()
    async with Session() as session:
        rows = (await session.execute(
            select(User.user_id, User.username, User.first_name, User.last_name, Attendance.clicked_at)
            .join(Attendance, Attendance.user_id == User.id)
            .where(Attendance.post_id == post_id)
            .order_by(Attendance.clicked_at)
        )).all()
        # برای نمایش header به channel_message_id نیاز داریم
        post = (await session.execute(select(Post).where(Post.id == post_id))).scalar_one()

    table_html, page, total_pages = _render_stats_table(rows, page=page, per_page=ROWS_PER_PAGE)
    header = f"آمار پست #{post.channel_message_id} — کل: {len(rows)}\n"
    await q.edit_message_text(
        header + table_html,
        parse_mode=ParseMode.HTML,
        reply_markup=_stats_kb(post_id, page, total_pages)
    )
    await q.answer()

def main() -> None:
    app = Application.builder().token(TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("post", cmd_post))
    app.add_handler(CommandHandler("export", cmd_export))
    app.add_handler(CommandHandler("stats", cmd_stats))

    # Button handlers
    app.add_handler(CallbackQueryHandler(on_button, pattern=r"^present$"))
    app.add_handler(CallbackQueryHandler(on_pick_channel, pattern=r"^(pickch:|chpage:)"))

    # Chat membership & channel posts (for logging and relationship capture)
    app.add_handler(ChatMemberHandler(on_my_chat_member, ChatMemberHandler.MY_CHAT_MEMBER))
    app.add_handler(MessageHandler(filters.ChatType.CHANNEL, on_channel_post))

    app.add_handler(CallbackQueryHandler(on_stats_page, pattern=r"^statspg:\d+:\d+$"))

    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
