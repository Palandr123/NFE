from pathlib import Path
# for modules imports
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

import telebot
from config import token
from inference import *
from telebot import custom_filters, types
from telebot.types import InlineKeyboardButton, InlineKeyboardMarkup

"""
Init for NFE ML model components
"""

model = NFEModel(PROJECT_ROOT)


"""
Init telegram bot
"""

bot = telebot.TeleBot(token, parse_mode="HTML")
bot.remove_webhook()

user_data = {}

"""
Telegram bot functions
"""


# def gen_markup():
#     markup = InlineKeyboardMarkup()
#     markup.row_width = 2
#     markup.add(
#         InlineKeyboardButton("Use my image", callback_data="cb_upload"),
#         InlineKeyboardButton("Generate image", callback_data="cb_generate")
#     )
#     return markup
#
# @bot.callback_query_handler(func=lambda call: True)
# def callback_query(call):
#     if call.data == "cb_upload":
#         # bot.answer_callback_query(call.id, "Upload my own image!!")
#         bot.send_message(call.from_user.id, "Please upload your image for editing.")
#     elif call.data == "cb_generate":
#         bot.send_message(call.from_user.id, "Okay, here is your generated image:")
#         # bot.answer_callback_query(call.id, "Generate image for me!!")

"""
Perform GAN inversion on photo sent
"""
@bot.message_handler(content_types=['photo'])
def upload_and_inversion(message):
    # get photo from user
    fileID = message.photo[-1].file_id
    file_info = bot.get_file(fileID)
    downloaded_file = bot.download_file(file_info.file_path)
    user_image_path = f'{message.chat.id}-user-image.png'
    gan_image_path = f'{message.chat.id}-gan-image.png'
    with open(user_image_path, 'wb') as new_file:
        new_file.write(downloaded_file)

    # apply gan inversion - very expensive, only on gpu!!
    z = model.gan_inversion(user_image_path)
    # fill user informatoin
    user_data[message.chat.id] = {}
    user_data[message.chat.id]['latent_vector'] = z.cpu().detach().numpy()
    model.generate_and_save_image(z, gan_image_path)
    user_data[message.chat.id]['gan_image_path'] = gan_image_path
    user_data[message.chat.id]['original_image_path'] = user_image_path
    # generate reply for user
    pics = model.picture_array(
        [user_data[message.chat.id]['original_image_path'], user_data[message.chat.id]['gan_image_path']]
    )
    medias_typed = [types.InputMediaPhoto(x) for x in pics]
    medias_typed[0].caption = 'Initial face and Face after GAN inversion!'
    # send before/after gan inversion images
    bot.send_media_group(message.chat.id, medias_typed)
    bot.send_message(message.chat.id, 'Now send /change to display possible changes!')

# @bot.message_handler(commands=['start'])
# def message_handler(message):
#     bot.send_message(message.chat.id, "Yes/no?", reply_markup=gen_markup())

"""
Help message.
TODO: populate with other commands
"""
@bot.message_handler(commands=['help'])
def send_welcome(message):
    bot.reply_to(message, 'Hello! This is Neural Face Editor bot!\nSend /face to generate random faces with GAN!')

"""
Generate random face with GAN.
"""
@bot.message_handler(commands=['face'])
def generate_face(message):
    # new image path, user-specific
    gan_image_path = f'{message.chat.id}-gan-image.png'
    # generate random latent code and generate image
    z = model.generate_and_save_random_image(gan_image_path)
    # fill user data
    user_data[message.chat.id] = {}
    user_data[message.chat.id]['latent_vector'] = z.cpu().detach().numpy()
    user_data[message.chat.id]['gan_image_path'] = gan_image_path
    user_data[message.chat.id]['original_image_path'] = None
    # send user his pic
    bot.send_photo(message.chat.id, photo=open(gan_image_path, 'rb'), caption='Your random face! Now send /change to display'
                                                                              'possible changes!')

"""
Latent space manipulation
TODO: Buttons
"""
@bot.message_handler(commands=['change'])
def change_face(message):
    try:
        # known vector
        z = user_data[message.chat.id]['latent_vector']
    except Exception:
        # generate new random image/vector
        bot.send_message(message.chat.id, 'No images yet! Generating random..')

        gan_image_path = f'{message.chat.id}-gan-image.png'
        z = model.generate_and_save_random_image(gan_image_path)
        user_data[message.chat.id] = {}
        user_data[message.chat.id]['latent_vector'] = z.cpu().detach().numpy()
        user_data[message.chat.id]['gan_image_path'] = gan_image_path
        user_data[message.chat.id]['original_image_path'] = None
        bot.send_photo(message.chat.id, photo=open(gan_image_path, 'rb'),
                       caption='Your random face! Now send /change again to display'
                               ' possible changes!')

    # parse input
    txt = message.text.split()
    feature = txt[1]
    value = txt[2]
    preserved_features = txt[3].split(',') if len(txt) > 3 else None
    z = user_data[message.chat.id]['latent_vector']
    # image manipulation
    medias = model.change_image(str(message.chat.id), z, feature, value, preserved_features)
    medias_typed = [types.InputMediaPhoto(x) for x in medias]
    medias_typed[0].caption = 'Your face changes! Try again with different parameters! /change'
    bot.send_media_group(message.chat.id, medias_typed)

# echo as default
@bot.message_handler(func=lambda x: True)
def echo_all_to_channel(message):
    bot.send_message(message.chat.id, message.text)

# start bot
bot.add_custom_filter(custom_filters.ChatFilter())
bot.infinity_polling()
