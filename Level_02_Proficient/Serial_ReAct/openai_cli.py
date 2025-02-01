# coding=utf-8
"""
@Author: Jacob Y
@Date  : 1/31/2025
@Desc  : ⚠️注：须创建环境变量"OPENAI_API_KEY_4_0"
"""
import os

from openai import OpenAI

APIKEY = os.environ['OPENAI_API_KEY_4_0']


def get_client():
    # print(APIKEY)
    return OpenAI(
                api_key=APIKEY,
                base_url='https://xiaoai.plus/v1'
            )
    