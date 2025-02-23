# coding=utf-8
"""
@Author: Jacob Y
@Date  : 2/3/2025
@Desc  : 
"""
import json
import random


def generate_more_training_samples():
    samples = []

    # 技术领域的正样本对
    tech_positive_pairs = [
        {
            "text_1": "深度学习模型在图像识别任务中表现出色",
            "text_2": "神经网络在处理计算机视觉问题时效果很好",
            "label": 1
        },
        {
            "text_1": "Python是一门简单易学的编程语言",
            "text_2": "Python作为入门编程语言非常友好",
            "label": 1
        },
        {
            "text_1": "区块链技术可以提供去中心化的解决方案",
            "text_2": "区块链为分布式系统提供了新的可能性",
            "label": 1
        }
    ]

    # 生活场景的正样本对
    life_positive_pairs = [
        {
            "text_1": "这家餐厅的川菜非常地道",
            "text_2": "这间店的四川菜味道很正宗",
            "label": 1
        },
        {
            "text_1": "周末去公园散步是很好的运动方式",
            "text_2": "在公园里慢走能够帮助保持健康",
            "label": 1
        },
        {
            "text_1": "这部电影的剧情非常感人",
            "text_2": "这个电影的故事情节很打动人",
            "label": 1
        }
    ]

    # 商业领域的正样本对
    business_positive_pairs = [
        {
            "text_1": "该公司今年的营收增长显著",
            "text_2": "这家企业的年度收入大幅提升",
            "label": 1
        },
        {
            "text_1": "电商平台在双十一期间销售火爆",
            "text_2": "购物节期间网上零售额创新高",
            "label": 1
        },
        {
            "text_1": "新能源汽车市场前景广阔",
            "text_2": "电动车行业发展潜力巨大",
            "label": 1
        }
    ]

    # 负样本对（跨领域）
    negative_pairs = [
        {
            "text_1": "人工智能正在改变我们的生活",
            "text_2": "今年的芒果特别甜",
            "label": -1
        },
        {
            "text_1": "这款相机的性能很专业",
            "text_2": "学习一门外语需要坚持",
            "label": -1
        },
        {
            "text_1": "股市今天表现不错",
            "text_2": "冬天要注意保暖",
            "label": -1
        },
        {
            "text_1": "这个季节适合种植草莓",
            "text_2": "量子计算机的发展突飞猛进",
            "label": -1
        },
        {
            "text_1": "新开的咖啡店环境很好",
            "text_2": "太空探索任务取得重大进展",
            "label": -1
        }
    ]

    # 教育领域的正样本对
    education_positive_pairs = [
        {
            "text_1": "在线教育为学习提供了更多可能",
            "text_2": "远程教学平台让学习更加灵活",
            "label": 1
        },
        {
            "text_1": "良好的学习习惯对成长很重要",
            "text_2": "培养正确的学习方法很有必要",
            "label": 1
        }
    ]

    # 医疗健康领域的正样本对
    health_positive_pairs = [
        {
            "text_1": "规律运动对身体健康很重要",
            "text_2": "坚持锻炼能够提升身体素质",
            "label": 1
        },
        {
            "text_1": "均衡的饮食有助于保持健康",
            "text_2": "科学的营养搭配对身体有好处",
            "label": 1
        }
    ]

    # 合并所有样本
    all_pairs = (tech_positive_pairs + life_positive_pairs +
                 business_positive_pairs + negative_pairs +
                 education_positive_pairs + health_positive_pairs)

    # 随机打乱样本顺序
    random.shuffle(all_pairs)

    # 转换为训练数据格式
    for pair in all_pairs:
        samples.append({
            "text": [pair["text_1"], pair["text_2"]],
            "label": pair["label"]
        })

    return samples


def main():
    training_data = generate_more_training_samples()

    # 保存为JSON文件
    with open('../data/train_data.json', 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

    print(f"已生成 {len(training_data)} 个训练样本")


if __name__ == "__main__":
    main()