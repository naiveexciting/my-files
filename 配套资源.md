
技术路线：端侧 YOLOv8（视觉识别） + 扣子 Coze（RAG智能体与多轮对话）文件组成

文件组成：项目说明手册、扣子(Coze)智能体搭建指南、环境依赖表、模型训练脚本、核心算法模块、交互程序代码。

#### 第一部分：项目说明与使用手册

1. **项目简介**

本项目旨在解决中学生物与地理科学观测课堂中“看不见、问不着、等不及”的教学痛点。我们构建了一套“多模态AI智能观测助手”，通过统一的技术底座，无缝接入显微镜与望远镜的数字图像源，实现：

(1) 微观细胞分裂期识别（前期、中期、后期、末期等）；

(2) 宏观月面地貌识别（环形山、月海、辐射纹等）；

(3) 图像质量智能评测（亮度曝光、清晰度焦距异常提示）；

(4) 基于扣子(Coze)智能体的互动引导（通过调用我们在扣子平台搭建的专属智能体API，结合学科知识库，实现启发式反问）。

2. **系统架构设计**

系统采用“端云协同”的轻量化架构，降低了学校硬件部署门槛：

(1) 硬件采集层：支持显微镜电子目镜或智能终端摄像头采集画面。

(2) 端侧视觉层 (YOLOv8)：在本地电脑实现秒级高精度目标框选与分类（不依赖云端，保障图像隐私与实时性）。

(3) 云端智能体层 (扣子 Coze)：在字节跳动“扣子”平台上搭建专属教育智能体，集成《高中生物》《高中地理》教参作为知识库（RAG），处理复杂逻辑。

(4) 交互应用层 (Streamlit)：将本地视觉结果作为“上下文”，通过 API 传递给扣子智能体，构建出双栏比对的 Web 交互环境。

3. **环境安装与快速部署指南**

第一步：环境准备

conda create -n science_ai python=3.9 -y

conda activate science_ai

第二步：安装依赖

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

第三步：配置扣子(Coze)密钥

在项目根目录下新建一个.env文件，填入你在扣子平台发布的 API Token 和 Bot ID：

COZE_API_TOKEN=pat_xxxxxxxxxxxxxxxxxxxxxx # 你的扣子个人访问令牌

COZE_BOT_ID=735xxxxxxxxxxxxxxxx # 你发布的智能体 ID

第四步：启动服务

确保best.pt（YOLO视觉权重）在同级目录，运行：

streamlit run app.py

#### 第二部分：扣子 (Coze) 智能体搭建指南

1. **智能体基础设置**

(1) 智能体名称：科学观测引导助教（细胞与星辰版）

(2) 模型选择：豆包·Pro（推荐，具备较强的逻辑推理与学科基础）。

2. **提示词与人设**

在扣子的“人设与回复逻辑”编排框中，填入以下严格的提示词，角色：你是一名专业、耐心且循循善诱的中学生物与地理科学观测助教；任务：学生正在使用显微镜或望远镜进行观测。前端系统会将【当前视觉AI识别到的目标标签】以及【学生的提问】发送给你。你需要结合这两者，并检索你的专属知识库，为学生提供解答，并引导他们继续探究。如果前端传来的视觉特征是“未发现预设特征”，请提示学生先调整显微镜焦距或寻找典型区域，语气需亲切、温和、鼓励，符合中学生认知水平。

3. **核心工作流**

(1) 分析前端传来的信息格式：前端通常会发送如“[当前视野识别特征：前期细胞] 学生提问：这是什么？”的信息。

(2) 知识检索：优先从你的知识库中检索与“识别特征”和“提问”相关的中学教材内容。

(3) 组织回答：绝不长篇大论，字数控制在 150 字以内、优先使用初高中课程标准中的规范名词、禁止直接把最终结论全部告诉学生。

(4) 启发反问：必须在回答的最后，另起一行并加粗标出“�� 助教反问：”，追加1个相关的启发性问题（如要求学生对比形态、推测成因或寻找下一个分裂期目标），引导学生继续观察。

4. **知识库挂载**

(1) 在扣子左侧栏选择“知识库” -> “添加知识库”。

(2) 生物数据集：上传《高中生物必修一（分子与细胞）》中关于“观察根尖分生区组织细胞的有丝分裂”的实验指导教参 PDF。

(3) 地理数据集：上传《高中地理》中关于月相与月面地貌（环形山、月海）的科普拓展材料 PDF。

5. 发布与 API 授权

(1) 点击右上角“发布”，选择发布渠道为 “Bot API”。

(2) 在个人空间 -> “扣子 API” 中生成 Personal Access Token (PAT)。

(3) 记录 URL 中的 Bot ID（形如 735...）。将这两者填入本地项目的.env配置文件中。

#### 第三部分：环境依赖配置 (requirements.txt)

# 计算机视觉与图像处理库

ultralytics>=8.0.20 # YOLOv8 官方框架，用于端侧目标检测

opencv-python>=4.8.0 # 图像处理（亮度、拉普拉斯清晰度计算等）

numpy>=1.24.0 # 基础矩阵与数值计算

Pillow>=10.0.0 # 图像读取与格式转换

# Web 交互与网络请求

streamlit>=1.30.0 # 快速构建数据应用的前端框架

requests>=2.31.0 # 用于向扣子(Coze)发送 API 请求

python-dotenv>=1.0.0 # 环境变量管理，用于安全加载扣子 Token

#### 第四部分：核心源代码

**1. 目标检测模型训练脚本**

端侧视觉底座，使用学生真实观测数据训练。

# -*- coding: utf-8 -*-

"""

从细胞到星辰 - 视觉目标检测模型训练脚本

"""

import os

from ultralytics import YOLO

def main():

    print("开始初始化多模态科学观测模型训练流水线...")

    # 加载预训练模型

    model = YOLO("yolov8n.pt")

    yaml_path = "science_observe.yaml"

    if not os.path.exists(yaml_path):

        raise FileNotFoundError(f"未找到数据集配置文件 {yaml_path}！")

    print("启动训练...")

    model.train(

        data=yaml_path,

        epochs=150,                # 训练150轮

        imgsz=640,                 # 图像缩放

        batch=16,                  

        device=0,                  # GPU加速

        patience=30,               # 早停机制防止过拟合

        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, # 色调增强模拟显微镜光源变化

        project="runs/science_observe",

        name="multimodal_exp01"

    )

    # 导出模型为通用 ONNX 格式

    model.export(format="onnx", simplify=True)

    print("训练完成！权重已保存。")

if __name__ == "__main__":

    main()

**2. 核心算法处理与扣子API通讯模块**

该模块负责图像质量检测、目标识别，并将识别结果作为上下文打包，调用扣子(Coze)的智能体 API。

# -*- coding: utf-8 -*-

"""

从细胞到星辰 - 核心 AI 引擎 (core_ai.py)

整合本地 YOLOv8 视觉识别与云端 Coze(扣子) 智能体 API

"""

import cv2

import os

import requests

import json

from ultralytics import YOLO

from dotenv import load_dotenv

# 加载环境变量中的扣子配置

load_dotenv()

COZE_API_TOKEN = os.getenv("COZE_API_TOKEN", "")

COZE_BOT_ID = os.getenv("COZE_BOT_ID", "")

class ScienceObservationAI:

    def __init__(self, model_path="best.pt"):

        """

        初始化智能体基座：加载视觉模型

        """

        # 视觉检测模块初始化

        try:

            self.detector = YOLO(model_path)

            self.model_loaded = True

            print("[INFO] 本地 YOLOv8 视觉检测模型加载成功！")

        except Exception as e:

            print(f"[ERROR] 视觉模型加载失败: {e}")

            self.model_loaded = False

    def check_image_quality(self, frame_cv2):

        """

        图像质量智能侦测：防误判与操作规范提示

        """

        gray = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2GRAY)

        brightness = gray.mean()

        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        if brightness < 35:

            return False, "图像偏暗：请检查显微镜反光镜/光源，或增大望远镜曝光度。"

        if brightness > 220:

            return False, "图像过曝：光线太强，请调小光圈。"

        if sharpness < 50:

            return False, "视野模糊：未准确对焦，请慢慢转动细准焦螺旋。"

        return True, "图像质量优良，可进行精准识别。"

    def detect_objects(self, frame_cv2, conf_threshold=0.35):

        """

        执行视觉目标识别

        """

        if not self.model_loaded:

            return frame_cv2, []

        results = self.detector(frame_cv2, conf=conf_threshold)

        result = results[0]

        labels = [self.detector.names[int(box.cls)] for box in result.boxes]

        annotated_frame = result.plot() # 画框

        unique_labels = list(set(labels))

        return annotated_frame, unique_labels

    def generate_answer_via_coze(self, labels, question, user_id="student_01"):

        """

        调用扣子 (Coze) Bot API 获取智能解答与反问

        文档: [https://www.coze.cn/docs/developer_guides/chat_v3](https://www.coze.cn/docs/developer_guides/chat_v3)

        """

        if not COZE_API_TOKEN or not COZE_BOT_ID:

            return "系统配置错误：未配置扣子 (Coze) API 令牌或 Bot ID。"

        labels_str = "、".join(labels) if labels else "未发现预设特征"

        # 构造发给扣子的融合提示词 (视觉上下文 + 学生问题)

        combined_prompt = f"[当前视野机器视觉识别到的特征]：{labels_str}\n[学生提问]：{question}"

        url = '[https://api.coze.cn/v3/chat](https://api.coze.cn/v3/chat)'

        headers = {

            'Authorization': f'Bearer {COZE_API_TOKEN}',

            'Content-Type': 'application/json'

        }

        payload = {

            "bot_id": COZE_BOT_ID,

            "user_id": user_id,

            "stream": False,

            "auto_save_history": True,

            "additional_messages": [

                {

                    "role": "user",

                    "content": combined_prompt,

                    "content_type": "text"

                }

            ]

        }

        try:

            # 发起 API 请求

            response = requests.post(url, headers=headers, json=payload, timeout=15)

            response.raise_for_status()

            data = response.json()

            # 解析扣子 V3 API 的返回结构 (取 type == 'answer' 的消息)

            if data.get("code") == 0 and "data" in data:

                messages = data["data"]

                # 此处简单起见，取扣子的第一条非结束类的回复文本（实际可根据需要解析具体结构）

                # 注意：实际Coze v3 API是异步的，需要建立循环查询 status，此处代码为简化示意版

                # 如果是简化版，可以直接用 Coze Chat v2 API，逻辑更简单

                return self._call_coze_v2_fallback(combined_prompt, user_id)

            else:

                return f"扣子 API 业务报错：{data.get('msg')}"

        except Exception as e:

            return f"无法连接到扣子云端服务器，请检查网络。报错信息：{str(e)}"

    def _call_coze_v2_fallback(self, combined_prompt, user_id):

        """

        使用 Coze V2 同步接口（便于流式与非流式调用展示）

        """

        url = '[https://api.coze.cn/open_api/v2/chat](https://api.coze.cn/open_api/v2/chat)'

        headers = {

            'Authorization': f'Bearer {COZE_API_TOKEN}',

            'Content-Type': 'application/json',

            'Connection': 'keep-alive'

        }

        payload = {

            "bot_id": COZE_BOT_ID,

            "user": user_id,

            "query": combined_prompt,

            "stream": False

        }

        resp = requests.post(url, headers=headers, json=payload, timeout=30)

        data = resp.json()

        if data.get("code") == 0:

            # 提取 assistant 的文本回答

            for msg in data.get("messages", []):

                if msg.get("type") == "answer":

                    return msg.get("content")

        return f"扣子API返回异常: {data.get('msg')}"

**3. 可视化 Web 前端主程序**

使用 Streamlit 渲染界面，衔接本地视觉与云端智能体。

# -*- coding: utf-8 -*-

"""

从细胞到星辰 - 交互式可视化教学终端

"""

import streamlit as st

import cv2

import numpy as np

from PIL import Image

from core_ai import ScienceObservationAI

import time

import uuid

# 页面配置

st.set_page_config(page_title="多模态AI观测助手", layout="wide")

# 加载本地模型基座

@st.cache_resource

def load_ai_assistant():

    return ScienceObservationAI()

ai_agent = load_ai_assistant()

# 初始化 Session State

if "current_labels" not in st.session_state:

    st.session_state.current_labels = []

if "messages" not in st.session_state:

    st.session_state.messages = [{"role": "assistant", "content": "你好！我是你通过【扣子】搭建的专属科学观测助教。请在左侧上传画面，然后向我提问吧！"}]

if "session_id" not in st.session_state:

    st.session_state.session_id = str(uuid.uuid4()) # 用于保持扣子多轮对话记忆

with st.sidebar:

    st.title("系统参数设置")

    st.info("架构：本地 YOLOv8 视觉 + 扣子(Coze) RAG智能体")

    conf_threshold = st.slider("视觉识别灵敏度", 0.1, 0.9, 0.35, 0.05)

st.title("从细胞到星辰：多模态AI智能观测助手")

st.divider()

col_left, col_right = st.columns([1.2, 1])

# 左侧：端侧视觉处理区

with col_left:

    st.subheader("实时观测视窗 (端侧处理)")

    uploaded_file = st.file_uploader("上传当前视野截屏 (支持 jpg/png)", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:

        image_pil = Image.open(uploaded_file).convert("RGB")

        frame_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        # 质量评估

        is_good, quality_msg = ai_agent.check_image_quality(frame_cv2)

        if not is_good:

            st.error(quality_msg)

        else:

            st.success(quality_msg)

        # 本地视觉识别

        with st.spinner('YOLOv8 正在本地扫描特征...'):

            annotated_frame, labels = ai_agent.detect_objects(frame_cv2, conf_threshold=conf_threshold)

        st.session_state.current_labels = labels

        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        st.image(annotated_rgb, caption="AI 智能视野重构图", use_column_width=True)

        st.markdown("**发现目标**：" + "  ".join([f"`{label}`" for label in labels]) if labels else "**发现目标**：`未检测到明显特征`")

# 右侧：扣子智能体交互区

with col_right:

    st.subheader("Coze 启发式交流舱 (云端处理)")

    chat_container = st.container(height=500)

    with chat_container:

        for msg in st.session_state.messages:

            avatar = "" if msg["role"] == "assistant" else ""

            with st.chat_message(msg["role"], avatar=avatar):

                st.markdown(msg["content"])

    question = st.chat_input("向扣子智能体提问：你能说出中期的特征吗？")

    if question:

        st.session_state.messages.append({"role": "user", "content": question})

        with chat_container:

            with st.chat_message("user", avatar=""):

                st.markdown(question)

        with chat_container:

            with st.chat_message("assistant", avatar=""):

                with st.spinner("扣子智能体检索知识库思考中..."):

                    labels = st.session_state.current_labels

                    # 发送给扣子平台

                    answer = ai_agent.generate_answer_via_coze(labels, question, user_id=st.session_state.session_id)

                    st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})