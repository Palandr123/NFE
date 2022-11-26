import locale
import telebot
from telebot import custom_filters
from inference import *

# locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8')
bot = telebot.TeleBot("secret_token", parse_mode="HTML")

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, 'Hello! This is Neural Face Editor bot!\nSend /face to generate random faces with GAN!')


@bot.message_handler(commands=['face'])
def generate_face(message):
    imgpath, _ = generate_and_save_random_image(G)
    bot.send_photo(message.chat.id, photo=open(imgpath, 'rb'), caption='Your random face!')


@bot.message_handler(commands=['change'])
def change_face(message):
    imgpath, z = generate_and_save_random_image(G)
    txt = message.text.split()
    feature = txt[1]
    value = txt[2]
    preserved_features = txt[3].split(',') if len(txt) > 3 else None
    bot.send_photo(message.chat.id, photo=open(imgpath, 'rb'), caption='Your random face!')
    medias = change_image(G, z.numpy(), feature, value, preserved_features)
    medias_typed = [types.InputMediaPhoto(x) for x in medias]
    medias_typed[0].caption = 'Your face changes!'
    bot.send_media_group(message.chat.id, medias_typed)

@bot.message_handler(func=lambda x: True)
def echo_all_to_channel(message):
    bot.send_message(message.chat.id, message.text)


device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
G = torch.load("data/Generator_v2_150.pth", map_location=device)
bot.add_custom_filter(custom_filters.ChatFilter())
bot.infinity_polling()
