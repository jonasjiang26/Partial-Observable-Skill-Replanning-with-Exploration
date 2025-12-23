from openai import OpenAI
import base64
import json

client = OpenAI()

# 读取图片并转换为 base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

agentview_image_base64 = encode_image("/home/jing/图片/prompt_test_agentview.png")
ee_image_base64 = encode_image("/home/jing/图片/prompt_test_ee.png")

# 读取 JSON 文件内容
with open("/home/jing/POSRE/prompt_test_spatial_relation1.json", "r", encoding="utf-8") as f:
    spatial_relation_content1 = f.read()
with open("/home/jing/POSRE/prompt_test_spatial_relation2.json", "r", encoding="utf-8") as f:
    spatial_relation_content2 = f.read()

# 构建提示词
prompt_text1 = f"""You are an AI assistant guiding a single-arm robot to table tasks: the robot arm can interact with all the items on the table. You will receive a command from human and two images from two cameras: one for a global view and one on the robot wrist for detailed views. Beside this, you will receive a description of the spatial relation of the items in JSON format. Your task: plan the sequence of low level instructions that are to be executed by VLA. based on the given command, images, and the given spatial relation. Example sequence: "open the fridge, pick up the milk, put milk in the fridge, close the fridge". Do not use the object id from the spatial relation, please use the normal name of the item in the instruction sequence.

Spatial Relations (JSON):
{spatial_relation_content1}

Command: "Put all the dairy products in the fridge."
"""

prompt_text2 = f"""Now one instruction is executed, the rest of the instruction sequence is:
4. Close the fridge.
Also the spatial relation is updated. The new Spatial relation:
Spatial Relations (JSON):
{spatial_relation_content2}

Your Task: check if the rest instructions need to be re-planed. If yes, give me the re-planed sequence. If no, give me the unchanged rest sequence.
"""

# 初始化消息历史列表
messages_history = []

# 第一条消息（包含图片和提示词）
first_message = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": prompt_text1
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{agentview_image_base64}"
            }
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{ee_image_base64}"
            }
        }
    ]
}

# 将第一条消息添加到历史
messages_history.append(first_message)

# 第一次 API 调用
response1 = client.chat.completions.create(
    model="gpt-4o",  # 或 "gpt-4-vision-preview" 等支持视觉的模型
    messages=messages_history
)

# 获取第一次响应
first_response_content = response1.choices[0].message.content
print(first_response_content)

# 将助手的响应添加到消息历史
messages_history.append({
    "role": "assistant",
    "content": first_response_content
})

# 第二条消息（在同一个聊天会话中）
second_message = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": prompt_text2
        }
    ]
}

# 将第二条消息添加到历史
messages_history.append(second_message)

# 第二次 API 调用（包含完整的对话历史）
response2 = client.chat.completions.create(
    model="gpt-4o",
    messages=messages_history
)

# 获取第二次响应
second_response_content = response2.choices[0].message.content
print("第二次响应:")
print(second_response_content)

# 如果需要继续对话，可以继续添加消息到 messages_history