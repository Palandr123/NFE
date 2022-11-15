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


@bot.message_handler(commands=['face'])
def generate_face(message):
    img = inference()
    imgpath = 'image.png'
    plt.imsave(imgpath, torch.clip(img, 0, 1).permute([1, 2, 0]).detach().cpu().numpy())
    bot.send_photo(message.chat.id, photo=open(imgpath, 'rb'), caption='Your random face!')


@bot.message_handler(func=lambda x: True)
def echo_all_to_channel(message):
    bot.send_message(message.chat.id, message.text)


bot.add_custom_filter(custom_filters.ChatFilter())
bot.infinity_polling()
