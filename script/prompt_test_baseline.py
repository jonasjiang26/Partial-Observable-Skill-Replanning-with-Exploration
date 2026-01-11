from openai import OpenAI
import base64
import json

client = OpenAI()

# 读取图片并转换为 base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

agentview_image_base64 = encode_image("/home/jing/图片/prompt_test_movie_night/init_agentview.png")
ee_image_base64 = encode_image("/home/jing/图片/prompt_test_movie_night/init_ee.png")
agentview_image_middle_base64 = encode_image("/home/jing/图片/prompt_test_movie_night/opened_agentview.png")
ee_image_middle_base64 = encode_image("/home/jing/图片/prompt_test_movie_night/opened_ee.png")

# 构建提示词
prompt_text1 = f"""
Command: "It is movie night, make me a snack."
"""

prompt_text2 = f"""Now the skill: "open the microwave" is executed,

Your Task: check if the rest instructions need to be re-planed. If yes, give me the re-planed sequence. If no, give me the unchanged rest sequence
"""

# 初始化消息历史列表
messages_history = []

# System 消息：定义 AI 的角色和基本行为
system_message = {
    "role": "system",
    "content": """You are an AI assistant guiding a single-arm robot to table tasks. 
The robot arm can interact with all the items on the table. 
You will receive commands from humans and images from two cameras (one for a global view and one on the robot wrist for detailed views), 
Your tasks: 
1. Based on the given command and images, evaluate the feasibility of the command execution. 
   If it is not feasible to execute the command, tell me why. 
   If it is feasible, proceed to the next task.
2. Plan the sequence of skills that are to be executed by low-level policy based on the given command and images.

Important rules:
- Always replace "the object" with the actual name of the item
- Example sequence: "open the fridge, pick up the milk, put milk in the fridge, close the fridge"."""
}

# 将 system 消息添加到历史（放在最前面）
messages_history.append(system_message)

# 第一条消息（包含图片和具体任务）
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
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{agentview_image_middle_base64}"
            }
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{ee_image_middle_base64}"
            }
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
print("第二次回答:")
print(second_response_content)

# 如果需要继续对话，可以继续添加消息到 messages_history
