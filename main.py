import streamlit as st
from utils import dateframe_agent
import pandas as pd

def create_chart(input_data, chart_type):
    df_data = pd.DataFrame(input_data["data"], columns=input_data["columns"])
    df_data.set_index(input_data["columns"][0], inplace=True)
    if chart_type == "bar":
        st.bar_chart(df_data)
    elif chart_type == "line":
        st.line_chart(df_data)
    elif chart_type == "scatter":
        st.scatter_chart(df_data)

st.title("AI数据分析工具🔧")
with st.sidebar:
    openai_api_key = st.text_input("请输入OpenAI API密钥：",
                                    type="password",
                                    value="sk-sxe8evtSOHjvwW6M56FdD5270b8a4f1aBeE704Ff36310147")
with st.sidebar:
    st.markdown('[📑 AI智能PDF问答工具](https://platform.openai.com/account/api-keys)')

data=st.file_uploader("上传你的CSV文件：", type="csv")
if data:
    st.session_state["df"] = pd.read_csv(data)
    with st.expander("原始数据"):
        st.dataframe(st.session_state["df"])#把 data frame 作为变量储存进会话状态里。
query=st.text_area("请输入你关于以上表格的问题，或数据提取请求，或可视化要求（支持散点图、折线图、条形图）：")
button=st.button("提交")
# if button and not openai_api_key:
#      st.info("请输入你的OpenAI API密钥")
if button and 'df'not in st.session_state:
     st.info("请上传你的CSV文件")
if button and "df" in st.session_state:
    with st.spinner("正在分析中，请稍等..."):
         response_dict = dateframe_agent(st.session_state["df"], query,openai_api_key)
         if 'answer' in response_dict:
             st.write(response_dict['answer'])
         if 'table' in response_dict:
             st.table(pd.DataFrame(response_dict['table']["data"],
                                   columns=response_dict["table"]["columns"]))
         if "bar" in response_dict:
            create_chart(response_dict["bar"], "bar")
         if "line" in response_dict:
            create_chart(response_dict["line"], "line")
         if "scatter" in response_dict:
            create_chart(response_dict["scatter"], "scatter")
         
         
    
    