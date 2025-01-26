from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Load a smaller model, e.g., TinyLlama
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
# Function to start the bot
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! I'm your AI Assistant bot. How can I help you today?")

# Function to handle messages
async def process(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Relays the received message back to the user
    user_message = update.message.text
    #print(f"User message: {user_message}")  # Display on PC/laptop
    response = generate_response_from_llm(user_message)
    await update.message.reply_text(response)


def generate_response_from_llm(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")  # Convert input to tokens
    outputs = model.generate(inputs["input_ids"],
        max_length=300,           # Set an appropriate max length for responses
        num_return_sequences=1,  # Generate only one response
        no_repeat_ngram_size=2,  # Avoid repetitive phrases
        top_p=0.9,               # Use nucleus sampling for diversity
        temperature=1.0          # Control randomness (lower = more deterministic)
    )  # Generate response
    print("Input Tokens:", inputs)
    print("Output Tokens:", outputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    API_TOKEN = "..."   # Add your API token here (from BotFather)
    application = Application.builder().token(API_TOKEN).build()

    # Add handlers for commands and messages
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process))

    # Start polling for updates
    application.run_polling()

if __name__ == '__main__':
    main()

