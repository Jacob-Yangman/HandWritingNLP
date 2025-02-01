# coding=utf-8
"""
@Author: Jacob Y
@Date  : 1/31/2025
@Desc  : 5个可串行调用工具
"""
import random


# 心情与菜品推荐
def mood_based_dish(mood):
    mood_dish_mapping = {
        "happy": "宫保鸡丁",
        "sad": "酸辣土豆丝",
        "romantic": "红酒炖牛肉",
        "stressed": "麻辣火锅",
        "energetic": "烧烤"
    }

    dish = mood_dish_mapping.get(mood.lower(), None)
    return {"dish": dish} if dish else {"message": "未找到符合心情的菜品"}


# 菜品与鸡尾酒搭配
def dish_and_drink_pairing(dish):
    pairing = {
        "宫保鸡丁": "Martini",
        "酸辣土豆丝": "Margarita",
        "红酒炖牛肉": "Old Fashioned",
        "麻辣火锅": "Cosmopolitan",
        "烧烤": "Mojito"
    }

    drink = pairing.get(dish, None)
    return {"drink": drink} if drink else {"message": "未找到合适的鸡尾酒搭配"}


# 鸡尾酒与音乐搭配
def drink_and_music_pairing(drink):
    pairing = {
        "Martini": "Jazz",
        "Margarita": "Latin",
        "Old Fashioned": "Blues",
        "Cosmopolitan": "Pop",
        "Mojito": "Reggae"
    }

    music_genre = pairing.get(drink, None)
    return {"music_genre": music_genre} if music_genre else {"message": "未找到合适的音乐风格"}


# 音乐与活动推荐
def music_and_activity_suggestion(music_genre):
    activities = {
        "Jazz": "在高档餐厅用餐",
        "Latin": "跳萨尔萨舞",
        "Blues": "参加现场演奏会",
        "Pop": "与朋友聚会",
        "Reggae": "海滩放松"
    }

    activity = activities.get(music_genre, None)
    return {"activity": activity} if activity else {"message": "未找到合适的活动"}


# 活动与旅行推荐
def activity_and_travel_recommendation(activity):
    destinations = {
        "在高档餐厅用餐": "巴黎",
        "跳萨尔萨舞": "古巴哈瓦那",
        "参加现场演奏会": "纽约",
        "与朋友聚会": "东京",
        "海滩放松": "马尔代夫"
    }

    destination = destinations.get(activity, None)
    return {"destination": destination} if destination else {"message": "未找到合适的旅行目的地"}
