import telebot
import locale

from telebot import custom_filters
from inference import *
import matplotlib.pyplot as plt

locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8')
bot = telebot.TeleBot("secret_token", parse_mode="HTML")

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, 'Hello! This is Neural Face Editor bot!\nSend /face to generate random faces with GAN!')


@bot.message_handler(func=lambda x: True)
def echo_all_to_channel(message):
    bot.send_message(message.chat.id, message.text)


bot.add_custom_filter(custom_filters.ChatFilter())
bot.infinity_polling()
