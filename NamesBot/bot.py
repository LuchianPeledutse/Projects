import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Hi! I am a bot that generates name. Type the language you want and I will generate a name for you")



if __name__ == '__main__':
    application = ApplicationBuilder().token('8161126638:AAHz1K7bwwhI-pWlMxa6Y0m9vXc0s1F6a04').build()
    
    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)
    
    application.run_polling()

