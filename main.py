import os
import re
import asyncio
from datetime import datetime
from io import BytesIO

from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types.safety_types import HarmCategory, HarmBlockThreshold
from google.generativeai.types.generation_types import (
    StopCandidateException,
    BlockedPromptException,
)
import PIL.Image as load_image

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)
from telegram.error import NetworkError, BadRequest
from telegram.constants import ChatAction, ParseMode

# Load environment variables from .env file
load_dotenv()

# =====================================
# Google Gemini Model Configuration
# =====================================

# Disable all safety filters for the Gemini models
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
}

# Configure the Google Generative AI API with your API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the Gemini models
model = genai.GenerativeModel("gemini-1.5-flash", safety_settings=SAFETY_SETTINGS)
img_model = genai.GenerativeModel("gemini-pro-vision", safety_settings=SAFETY_SETTINGS)

# =====================================
# Telegram Bot Configuration
# =====================================

# Get authorized users from environment variables
_AUTHORIZED_USERS = [
    i.strip() for i in os.getenv("AUTHORIZED_USERS", "").split(",") if i.strip()
]

# Define states for conversation handlers
NEW_INSTRUCTIONS, SAVE_INSTRUCTIONS = range(2)
ADD_INSTRUCTIONS, APPEND_INSTRUCTIONS = range(2, 4)

# =====================================
# Custom Filters
# =====================================

class AuthorizedUserFilter(filters.UpdateFilter):
    """Filter to allow only authorized users to interact with the bot."""

    def filter(self, update: Update):
        """Check if the user is authorized."""
        if not _AUTHORIZED_USERS:  # If no authorized users are specified, allow all
            return True
        return (
                update.message.from_user.username in _AUTHORIZED_USERS
                or str(update.message.from_user.id) in _AUTHORIZED_USERS
        )


# Create filter instances
AuthFilter = AuthorizedUserFilter()
MessageFilter = AuthFilter & ~filters.COMMAND & filters.TEXT
PhotoFilter = AuthFilter & ~filters.COMMAND & filters.PHOTO

# =====================================
# Bot Handlers
# =====================================

def new_chat(context: ContextTypes.DEFAULT_TYPE):
    """Starts a new chat session with the model, loading instructions from file."""
    with open("all_you_need_to_say.txt", "r", encoding="utf-8") as file:
        clean_text = file.read()
    context.chat_data["chat"] = model.start_chat(
        history=[
            {"role": "user", "parts": [clean_text]},
            {"role": "model", "parts": ["Sure."]},
        ]
    )


async def start(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /start command, sending a welcome message."""
    user = update.effective_user
    message = f"Hi {user.mention_html()}!\n\nStart sending messages with me to generate a response.\n\nSend /new to start a new chat session."
    await update.message.reply_html(message)

    # Log user interaction
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open("user_log.txt", "a") as log_file:
            log_file.write(f"{now} - User {user.id} - {user.username} started the chat.\n")
            log_file.flush()
    except IOError as e:
        print(f"Failed to write to log file: {e}")


async def help_command(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /help command, sending a list of available commands."""
    help_text = """
Basic commands:
/start - Start the bot
/help - Get help. Shows this message

Chat commands:
/new - Start a new chat session (model will forget previously generated messages)
/new_instructions - Replace the bot's instructions with new ones
/add_instructions - Appends additional instructions

Send a message to the bot to generate a response.
"""
    await update.message.reply_text(help_text)


async def newchat_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /new command, starting a new chat session."""
    init_msg = await update.message.reply_text(
        text="Starting new chat session...",
        reply_to_message_id=update.message.message_id,
    )
    new_chat(context)
    await init_msg.edit_text("New chat session started.")


# =====================================
# Instruction Management Handlers
# =====================================

async def new_instructions_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles the /new_instructions command, prompting for new instructions."""
    await update.message.reply_text("Please send me your new instructions.")
    return NEW_INSTRUCTIONS


async def save_instructions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Saves the new instructions to the file, overwriting previous content."""
    new_instructions = update.message.text
    try:
        with open("all_you_need_to_say.txt", "w", encoding="utf-8") as file:
            file.write(new_instructions)
        await update.message.reply_text("Instructions saved successfully!")
    except IOError as e:
        await update.message.reply_text(f"Error saving instructions: {e}")
    return ConversationHandler.END


async def add_instructions_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles the /add_instructions command, prompting for instructions to append."""
    await update.message.reply_text("Please send me the instructions you want to add.")
    return ADD_INSTRUCTIONS


async def append_instructions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Appends the additional instructions to the file."""
    additional_instructions = update.message.text
    try:
        with open("all_you_need_to_say.txt", "a", encoding="utf-8") as file:
            file.write("\n" + additional_instructions)
        await update.message.reply_text("Instructions added successfully!")
    except IOError as e:
        await update.message.reply_text(f"Error adding instructions: {e}")
    return ConversationHandler.END


# =====================================
# Message and Image Handling
# =====================================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles incoming text messages, sending them to the Gemini model for a response."""
    if context.chat_data.get("chat") is None:
        new_chat(context)

    text = update.message.text
    init_msg = await update.message.reply_text(
        text="Generating...", reply_to_message_id=update.message.message_id
    )
    await update.message.chat.send_action(ChatAction.TYPING)

    chat = context.chat_data.get("chat")
    response = None
    try:
        response = await chat.send_message_async(text, stream=True)
    except StopCandidateException as sce:
        print("Prompt: ", text, " was stopped. User: ", update.message.from_user)
        print(sce)
        await init_msg.edit_text("The model unexpectedly stopped generating.")
        chat.rewind()
        return
    except BlockedPromptException as bpe:
        print("Prompt: ", text, " was blocked. User: ", update.message.from_user)
        print(bpe)
        await init_msg.edit_text("Blocked due to safety concerns.")
        if response:
            await response.resolve()
        return

    full_plain_message = ""
    async for chunk in response:
        try:
            if chunk.text:
                full_plain_message += chunk.text
                message = format_message(full_plain_message)
                init_msg = await init_msg.edit_text(
                    text=message,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )
        except StopCandidateException:
            await init_msg.edit_text("The model unexpectedly stopped generating.")
            chat.rewind()
            continue
        except BadRequest:
            await response.resolve()
            continue
        except NetworkError:
            raise NetworkError("Looks like you're network is down. Please try again later.")
        except IndexError:
            await init_msg.reply_text(
                "Some index error occurred. This response is not supported."
            )
            await response.resolve()
            continue
        except Exception as e:
            print(e)
            if chunk.text:
                full_plain_message = chunk.text
                message = format_message(full_plain_message)
                init_msg = await update.message.reply_text(
                    text=message,
                    parse_mode=ParseMode.HTML,
                    reply_to_message_id=init_msg.message_id,
                    disable_web_page_preview=True,
                )
        await asyncio.sleep(0.1)  # Prevent rate limiting


async def handle_image(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles incoming images, sending them to the Gemini model for analysis."""
    init_msg = await update.message.reply_text(
        text="Generating...", reply_to_message_id=update.message.message_id
    )
    images = update.message.photo
    unique_images: dict = {}
    for img in images:
        file_id = img.file_id[:-7]
        if file_id not in unique_images:
            unique_images[file_id] = img
        elif img.file_size > unique_images[file_id].file_size:
            unique_images[file_id] = img
    file_list = list(unique_images.values())
    file = await file_list[0].get_file()
    a_img = load_image.open(BytesIO(await file.download_as_bytearray()))
    prompt = update.message.caption or "Analyse this image and generate response"
    response = await img_model.generate_content_async([prompt, a_img], stream=True)
    full_plain_message = ""
    async for chunk in response:
        try:
            if chunk.text:
                full_plain_message += chunk.text
                message = format_message(full_plain_message)
                init_msg = await init_msg.edit_text(
                    text=message,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )
        except StopCandidateException:
            await init_msg.edit_text("The model unexpectedly stopped generating.")
        except BadRequest:
            await response.resolve()
            continue
        except NetworkError:
            raise NetworkError("Looks like you're network is down. Please try again later.")
        except IndexError:
            await init_msg.reply_text(
                "Some index error occurred. This response is not supported."
            )
            await response.resolve()
            continue
        except Exception as e:
            print(e)
            if chunk.text:
                full_plain_message = chunk.text
                message = format_message(full_plain_message)
                init_msg = await update.message.reply_text(
                    text=message,
                    parse_mode=ParseMode.HTML,
                    reply_to_message_id=init_msg.message_id,
                    disable_web_page_preview=True,
                )

# =====================================
# Markdown to HTML Formatting
# =====================================

def escape_html(text: str) -> str: # This line
    """Escapes HTML special characters in a string."""
    text = text.replace("&", "&")
    text = text.replace("<", "<")
    text = text.replace(">", ">")
    return text

def apply_hand_points(text: str) -> str:
    """Replaces markdown bullet points (*) with right hand point emoji."""
    pattern = r"(?<=\n)\*\s(?!\*)|^\*\s(?!\*)"
    replaced_text = re.sub(pattern, "👉 ", text)
    return replaced_text

def apply_bold(text: str) -> str:
    """Replaces markdown bold formatting with HTML bold tags."""
    return re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)


def apply_italic(text: str) -> str:
    """Replaces markdown italic formatting with HTML italic tags."""
    return re.sub(r"(?<!\*)\*(?!\*)(?!\*\*)(.*?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)


def apply_code(text: str) -> str:
    """Replaces markdown code blocks with HTML <pre> tags."""
    return re.sub(r"```([\w]*?)\n([\s\S]*?)```", r"<pre lang='\1'>\2</pre>", text, flags=re.DOTALL)


def apply_monospace(text: str) -> str:
    """Replaces markdown monospace backticks with HTML <code> tags."""
    return re.sub(r"(?<!`)`(?!`)(.*?)(?<!`)`(?!`)", r"<code>\1</code>", text)


def apply_link(text: str) -> str:
    """Replaces markdown links with HTML anchor tags."""
    return re.sub(r"\[(.*?)\]\((.*?)\)", r'<a href="\2">\1</a>', text)


def apply_underline(text: str) -> str:
    """Replaces markdown underline with HTML underline tags."""
    return re.sub(r"__(.*?)__", r"<u>\1</u>", text)


def apply_strikethrough(text: str) -> str:
    """Replaces markdown strikethrough with HTML strikethrough tags."""
    return re.sub(r"~~(.*?)~~", r"<s>\1</s>", text)


def apply_header(text: str) -> str:
    """Replaces markdown header # with HTML header tags."""
    return re.sub(r"^(#{1,6})\s+(.*)", r"<b><u>\2</u></b>", text, flags=re.DOTALL)


def apply_exclude_code(text: str) -> str:
    """Applies text formatting to non-code lines."""
    lines = text.split("\n")
    in_code_block = False
    for i, line in enumerate(lines):
        if line.startswith("```"):
            in_code_block = not in_code_block
        if not in_code_block:
            formatted_line = lines[i]
            formatted_line = apply_header(formatted_line)
            formatted_line = apply_link(formatted_line)
            formatted_line = apply_bold(formatted_line)
            formatted_line = apply_italic(formatted_line)
            formatted_line = apply_underline(formatted_line)
            formatted_line = apply_strikethrough(formatted_line)
            formatted_line = apply_monospace(formatted_line)
            formatted_line = apply_hand_points(formatted_line)
            lines[i] = formatted_line
    return "\n".join(lines)


def format_message(text: str) -> str:
    """Formats the given message text from markdown to HTML."""
    formatted_text = escape_html(text)
    formatted_text = apply_exclude_code(formatted_text)
    formatted_text = apply_code(formatted_text)
    return formatted_text

# =====================================
# Bot Initialization
# =====================================

def start_bot() -> None:
    """Starts the Telegram bot."""
    application = Application.builder().token(os.getenv("BOT_TOKEN")).build()

    # Add command handlers
    application.add_handler(CommandHandler("start", start, filters=AuthFilter))
    application.add_handler(CommandHandler("help", help_command, filters=AuthFilter))
    application.add_handler(CommandHandler("new", newchat_command, filters=AuthFilter))

    # Add conversation handlers for instruction management
    new_instructions_handler = ConversationHandler(
        entry_points=[
            CommandHandler(
                "new_instructions", new_instructions_command, filters=AuthFilter
            )
        ],
        states={
            NEW_INSTRUCTIONS: [MessageHandler(filters.TEXT, save_instructions)],
        },
        fallbacks=[CommandHandler("cancel", start, filters=AuthFilter)],
    )
    add_instructions_handler = ConversationHandler(
        entry_points=[
            CommandHandler(
                "add_instructions", add_instructions_command, filters=AuthFilter
            )
        ],
        states={
            ADD_INSTRUCTIONS: [MessageHandler(filters.TEXT, append_instructions)],
        },
        fallbacks=[CommandHandler("cancel", start, filters=AuthFilter)],
    )
    application.add_handler(new_instructions_handler)
    application.add_handler(add_instructions_handler)

    # Add message and image handlers
    application.add_handler(MessageHandler(MessageFilter, handle_message))
    application.add_handler(MessageHandler(PhotoFilter, handle_image))

    # Start the bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    start_bot()
