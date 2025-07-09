import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import io
import json
import html
from telegram import Update, User, Message, Chat, Voice, PhotoSize, File, BotCommand
from telegram.constants import ParseMode, ChatAction
from telegram.ext import Application, ApplicationBuilder, CallbackContext
import openai

from bot.bot import (
    register_user_if_not_exists,
    is_bot_mentioned,
    start_handle,
    help_handle,
    help_group_chat_handle,
    retry_handle,
    message_handle,
    voice_message_handle,
    generate_image_handle,
    new_dialog_handle,
    cancel_handle,
    show_chat_modes_handle,
    settings_handle,
    show_balance_handle,
    edited_message_handle,
    error_handle,
    post_init,
    run_bot,
    split_text_into_chunks,
    HELP_MESSAGE
)

@pytest.fixture
def mock_db():
    with patch('bot.bot.db') as mock:
        mock.check_if_user_exists.return_value = False
        mock.get_user_attribute.return_value = None
        mock.get_dialog_messages.return_value = []
        yield mock

@pytest.fixture
def mock_context():
    context = MagicMock()
    context.bot = MagicMock()
    context.bot.username = "test_bot"
    context.bot.send_message = AsyncMock()
    context.bot.send_photo = AsyncMock()
    context.bot.send_voice = AsyncMock()
    context.bot.get_file = AsyncMock()
    context.bot.send_chat_action = AsyncMock()
    context.bot.edit_message_text = AsyncMock()
    return context

@pytest.fixture
def mock_user():
    return User(id=1, is_bot=False, first_name="Test", username="test_user")

@pytest.fixture
def mock_chat():
    return Chat(id=1, type="private")

@pytest.fixture
def mock_message(mock_user, mock_chat):
    message = Message(
        message_id=1,
        date=datetime.now(),
        chat=mock_chat,
        from_user=mock_user,
        text="test message"
    )
    return message

@pytest.fixture
def mock_update(mock_message):
    update = MagicMock(spec=Update)
    update.message = mock_message
    update.effective_chat = mock_message.chat
    update.effective_user = mock_message.from_user
    return update

@pytest.mark.asyncio
async def test_register_user_if_not_exists(mock_db, mock_update, mock_context):
    user = mock_update.message.from_user
    await register_user_if_not_exists(mock_update, mock_context, user)
    
    mock_db.check_if_user_exists.assert_called_once_with(user.id)
    mock_db.add_new_user.assert_called_once_with(
        user.id,
        mock_update.message.chat_id,
        username=user.username,
        first_name=user.first_name,
        last_name=user.last_name
    )

@pytest.mark.asyncio
async def test_is_bot_mentioned_private_chat(mock_update, mock_context):
    mock_update.message.chat.type = "private"
    result = await is_bot_mentioned(mock_update, mock_context)
    assert result is True

@pytest.mark.asyncio
async def test_is_bot_mentioned_group_chat_with_mention(mock_update, mock_context):
    mock_update.message.chat.type = "group"
    mock_update.message.text = f"@{mock_context.bot.username} test"
    result = await is_bot_mentioned(mock_update, mock_context)
    assert result is True

@pytest.mark.asyncio
async def test_start_handle(mock_db, mock_update, mock_context):
    await start_handle(mock_update, mock_context)
    
    mock_db.set_user_attribute.assert_called()
    mock_db.start_new_dialog.assert_called_once_with(mock_update.message.from_user.id)
    mock_update.message.reply_text.assert_called()

@pytest.mark.asyncio
async def test_help_handle(mock_db, mock_update, mock_context):
    await help_handle(mock_update, mock_context)
    
    mock_update.message.reply_text.assert_called_once_with(
        HELP_MESSAGE,
        parse_mode=ParseMode.HTML
    )

@pytest.mark.asyncio
async def test_retry_handle_no_messages(mock_db, mock_update, mock_context):
    mock_db.get_dialog_messages.return_value = []
    await retry_handle(mock_update, mock_context)
    
    mock_update.message.reply_text.assert_called_once_with(
        "No message to retry 🤷‍♂️"
    )

@pytest.mark.asyncio
async def test_new_dialog_handle(mock_db, mock_update, mock_context):
    await new_dialog_handle(mock_update, mock_context)
    
    mock_db.set_user_attribute.assert_called()
    mock_db.start_new_dialog.assert_called_once_with(mock_update.message.from_user.id)
    mock_update.message.reply_text.assert_called()

@pytest.mark.asyncio
async def test_cancel_handle_no_task(mock_db, mock_update, mock_context):
    await cancel_handle(mock_update, mock_context)
    
    mock_update.message.reply_text.assert_called_once_with(
        "<i>Nothing to cancel...</i>",
        parse_mode=ParseMode.HTML
    )

@pytest.mark.asyncio
async def test_edited_message_handle_private_chat(mock_update, mock_context):
    mock_update.edited_message = mock_update.message
    mock_update.edited_message.chat.type = "private"
    
    await edited_message_handle(mock_update, mock_context)
    
    mock_update.edited_message.reply_text.assert_called_once_with(
        "🤷 Unfortunately, message <b>editing</b> is not supported",
        parse_mode=ParseMode.HTML
    )

@pytest.mark.asyncio
async def test_error_handle(mock_update, mock_context):
    mock_context.error = Exception("Test error")
    await error_handle(mock_update, mock_context)
    
    mock_context.bot.send_message.assert_called()

@pytest.mark.asyncio
async def test_post_init():
    app = MagicMock(spec=Application)
    app.bot = MagicMock()
    app.bot.set_my_commands = AsyncMock()
    
    await post_init(app)
    
    app.bot.set_my_commands.assert_called_once()

def test_split_text_into_chunks():
    text = "12345"
    chunks = list(split_text_into_chunks(text, 2))
    assert chunks == ["12", "34", "5"]

@pytest.mark.asyncio
async def test_voice_message_handle(mock_db, mock_update, mock_context):
    mock_update.message.voice = Voice(
        file_id="test_file_id",
        duration=10,
        file_unique_id="test_unique_id"
    )
    
    mock_file = File(file_id="test_file_id", file_unique_id="test_unique_id")
    mock_context.bot.get_file.return_value = mock_file
    mock_file.download_to_memory = AsyncMock()
    
    with patch('bot.bot.openai_utils.transcribe_audio', new_callable=AsyncMock) as mock_transcribe:
        mock_transcribe.return_value = "transcribed text"
        await voice_message_handle(mock_update, mock_context)
        
        mock_transcribe.assert_called_once()
        mock_update.message.reply_text.assert_called()

@pytest.mark.asyncio
async def test_generate_image_handle(mock_db, mock_update, mock_context):
    with patch('bot.bot.openai_utils.generate_images', new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = ["http://test.image.url"]
        await generate_image_handle(mock_update, mock_context)
        
        mock_generate.assert_called_once()
        mock_update.message.chat.send_action.assert_called_with(action=ChatAction.UPLOAD_PHOTO)
        mock_update.message.reply_photo.assert_called()

@pytest.mark.asyncio
async def test_show_balance_handle(mock_db, mock_update, mock_context):
    mock_db.get_user_attribute.side_effect = [
        {"model1": {"n_input_tokens": 100, "n_output_tokens": 50}},
        10,  # n_generated_images
        20.0  # n_transcribed_seconds
    ]
    
    await show_balance_handle(mock_update, mock_context)
    
    mock_update.message.reply_text.assert_called_once()

@pytest.mark.asyncio
async def test_message_handle_empty_message(mock_db, mock_update, mock_context):
    mock_update.message.text = ""
    
    await message_handle(mock_update, mock_context)
    
    mock_update.message.reply_text.assert_called_with(
        "🤷 You sent <b>empty message</b>. Please, try again!",
        parse_mode=ParseMode.HTML
    )

@pytest.mark.asyncio
async def test_message_handle_with_photo(mock_db, mock_update, mock_context):
    mock_update.message.photo = [PhotoSize(file_id="test_id", file_unique_id="test_unique_id", width=100, height=100)]
    mock_file = File(file_id="test_id", file_unique_id="test_unique_id")
    mock_context.bot.get_file.return_value = mock_file
    mock_file.download_to_memory = AsyncMock()
    
    with patch('bot.bot._vision_message_handle_fn', new_callable=AsyncMock) as mock_vision:
        await message_handle(mock_update, mock_context)
        mock_vision.assert_called_once()

@pytest.mark.asyncio
async def test_show_chat_modes_handle(mock_db, mock_update, mock_context):
    await show_chat_modes_handle(mock_update, mock_context)
    
    mock_update.message.reply_text.assert_called_once()

@pytest.mark.asyncio
async def test_settings_handle(mock_db, mock_update, mock_context):
    await settings_handle(mock_update, mock_context)
    
    mock_update.message.reply_text.assert_called_once()