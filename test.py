### 시장별 종목 확인 ###



import requests as rq
from io import BytesIO
import numpy as np
import pandas as pd
import datetime
import os

# import dash
from dash import Dash, html, dcc, Input, Output, callback, dash_table
# from jupyter_dash import JupyterDash
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go





print(os.getcwd())
# os.chdir("/content/drive/MyDrive/test")


# today = (datetime.datetime.now() + datetime.timedelta(days=-0)).strftime("%Y%m%d")

today = datetime.datetime.now().strftime("%Y%m%d")
# today = datetime.datetime.now().strftime("%Y%m%d")
two_year_ago = (datetime.datetime.now() + datetime.timedelta(days=-365)).strftime("%Y%m%d")


#generate.cmd에 Request URL과 동일
gen_otp_url = "http://data.krx.co.kr/comm/fileDn/GenerateOTP/generate.cmd"



# 투자주의종목 확인
gen_otp_data1={
'locale': 'ko_KR',
'mktId': 'ALL',
'inqTpCd1': '01',
'inqTp': '1',
'tboxisuCd_finder_stkisu2_2': '전체',
'isuCd': 'ALL',
'isuCd2': 'ALL',
'codeNmisuCd_finder_stkisu2_2': '',
'param1isuCd_finder_stkisu2_2': 'ALL',
'strtDd': two_year_ago,
'endDd': today,
'inqCondTpCd': 'Y',
'share': '1',
'money': '1',
'csvxls_isNo': 'true',
'name': 'fileDown',
'url': 'dbms/MDC/STAT/issue/MDCSTAT23001'

}


# 투자경고종목 확인
gen_otp_data2={
'locale': 'ko_KR',
'mktId': 'ALL',
'tboxisuCd_finder_stkisu0_6': '전체',
'isuCd': 'ALL',
'isuCd2': 'ALL',
'codeNmisuCd_finder_stkisu0_6': '',
'param1isuCd_finder_stkisu0_6': 'ALL',
'strtDd': two_year_ago,
'endDd': today,
'inqCondTpCd': 'Y',
'share': '1',
'money': '1',
'csvxls_isNo': 'true',
'name': 'fileDown',
'url': 'dbms/MDC/STAT/issue/MDCSTAT23301'

}


# 투자위험종목 확인
gen_otp_data3={
'locale': 'ko_KR',
'mktId': 'ALL',
'tboxisuCd_finder_stkisu2_5': '전체',
'isuCd': 'ALL',
'isuCd2': 'ALL',
'codeNmisuCd_finder_stkisu2_5':'',
'param1isuCd_finder_stkisu2_5': 'ALL',
'strtDd': two_year_ago,
'endDd': today,
'inqCondTpCd': 'Y',
'share': '1',
'money': '1',
'csvxls_isNo': 'true',
'name': 'fileDown',
'url': 'dbms/MDC/STAT/issue/MDCSTAT23601'

}



# 종목코드 확인
gen_otp_data4={
'locale': 'ko_KR',
'mktId': 'ALL',
'share': '1',
'csvxls_isNo': 'false',
'name': 'fileDown',
'url': 'dbms/MDC/STAT/standard/MDCSTAT01901'
}


# 헤더 부분에 리퍼러(Refere)를 추가합니다
# 리퍼러란 링크를 통해서 각각의 웹사이트로 방문할 때 남는 흔적입니다. (로봇으로 인식을 하지 않게 하기 위함)
# generate.cmd - Headers - Request Headers 내 Referer

headers={"User-Agent": "Mozilla/5.0"}


otp1 = rq.post(gen_otp_url, gen_otp_data1,headers=headers).text
otp2 = rq.post(gen_otp_url, gen_otp_data2, headers = headers).text
otp3 = rq.post(gen_otp_url, gen_otp_data3, headers = headers).text
otp4 = rq.post(gen_otp_url, gen_otp_data4, headers = headers).text



# download.cmd에서 General의 Request URL 부분
down_url = 'http://data.krx.co.kr/comm/fileDn/download_csv/download.cmd'

# requests Module의 post함수를 이용하여 해당 url에 접속하여 otp 코드를 제출함
down1 = rq.post(down_url, {'code' : otp1}, headers=headers)
down2 = rq.post(down_url, {'code' : otp2}, headers=headers)
down3 = rq.post(down_url, {'code' : otp3}, headers=headers)
down4 = rq.post(down_url, {'code' : otp4}, headers=headers)




# 다운받은 csv파일을 pandas의 read_csv 함수를 이용하여 읽어 들임
# read_csv함수의 argument에 적합할 수 있도록 BytesIO함수를 이용하여 바이너 스트림 형태로 만든다.
caution = pd.read_csv(BytesIO(down1.content), encoding='EUC-KR')
caution_list = caution.iloc[:,[1,2,3,4,6,7,8,12]]
caution_list.insert(4,'해제일','-')
caution_list.insert(0,'시장경보','투자주의종목')
today_caution_list = caution_list[caution_list['지정일']==datetime.datetime.now().strftime("%Y/%m/%d")]
# caution = caution[caution['해제일'].isnull()]



warning = pd.read_csv(BytesIO(down2.content), encoding='EUC-KR')
warning_list = warning.iloc[:,[1,2,3,4,5,7,8,9,13]]
warning_list.insert(0,'시장경보','투자경고종목')
today_warning_list = warning_list[warning_list['해제일']=='-']



danger = pd.read_csv(BytesIO(down3.content), encoding='EUC-KR')
danger_list = danger.iloc[:,[1,2,3,4,5,7,8,9,13]]
danger_list.insert(0,'시장경보','투자위험종목')
danger_list.rename(columns={'지정일 익일_종가': '지정일 (D+1일)_종가'}, inplace=True)
today_danger_list = danger_list[danger_list['해제일']=='-']


code = pd.read_csv(BytesIO(down4.content), encoding='EUC-KR')
code = code[['표준코드','단축코드']]
code.columns = ['표준코드','종목코드']
code['종목코드'] = ('000'+code['종목코드'].apply(str)).apply(lambda x: x[-6:])

alert_list = pd.concat([caution_list, warning_list, danger_list])
alert_list['종목코드'] = ('000'+alert_list['종목코드'].apply(str)).apply(lambda x: x[-6:])
alert_list = alert_list.sort_values(by='지정일', ascending=False)
alert_list = alert_list.merge(code, on='종목코드', how='left')

today_alert_list = pd.concat([today_caution_list, today_warning_list, today_danger_list]).iloc[:,[0,1,2,3,4,5,6,7,8,9]]
today_alert_list['종목코드'] = ('000'+today_alert_list['종목코드'].apply(str)).apply(lambda x: x[-6:])
today_alert_list = today_alert_list.sort_values(by='지정일', ascending=False)



###############################################################################################


filter_alert_list = alert_list[alert_list['시장경보'].isin(['투자경고종목','투자위험종목'])]

filter_alert_list = filter_alert_list[filter_alert_list['시장구분'].isin(['KOSPI','KOSDAQ'])]

tmp_df = pd.DataFrame(columns=['시장경보','종목코드','종목명','시장구분','지정일','D-16','D-15','D-14','D-13','D-12','D-11','D-10','D-9','D-8','D-7','D-6','D-5','D-4','D-3','D-2','D-1','D-0','D+1','D+2','D+3','D+4','D+5','D+6','D+7','D+8','D+9','D+10','D+11','D+12','D+13','D+14','D+15'])




for i in range(filter_alert_list.shape[0]):

  tmp_df2 = pd.DataFrame(filter_alert_list.iloc[i,:5]).transpose()

  gen_otp_data1={
    'locale': 'ko_KR',
    'tboxisuCd_finder_stkisu0_2': filter_alert_list.iloc[i,1]+'/'+filter_alert_list.iloc[i,2],
    'isuCd': filter_alert_list.iloc[i,10],
    'isuCd2': filter_alert_list.iloc[i,1],
    'codeNmisuCd_finder_stkisu0_2': filter_alert_list.iloc[i,2],
    'param1isuCd_finder_stkisu0_2': 'ALL',
    'strtDd': (datetime.datetime.now() + datetime.timedelta(days=-730)).strftime("%Y%m%d"),
    'endDd': datetime.datetime.now().strftime("%Y%m%d"),
    'adjStkPrc_check': 'Y',
    'adjStkPrc': '2',
    'share': '1',
    'money': '1',
    'csvxls_isNo': 'false',
    'name': 'fileDown',
    'url': 'dbms/MDC/STAT/standard/MDCSTAT01701'

  }

  gen_otp_url = "http://data.krx.co.kr/comm/fileDn/GenerateOTP/generate.cmd"

  headers={"User-Agent": "Mozilla/5.0"}

  otp1 = rq.post(gen_otp_url, gen_otp_data1, headers = headers).text

  down_url = 'http://data.krx.co.kr/comm/fileDn/download_csv/download.cmd'

  down1 = rq.post(down_url, {'code' : otp1}, headers=headers)

  df = pd.read_csv(BytesIO(down1.content), encoding='EUC-KR')




  if tmp_df2['시장경보'].values=='투자위험종목':

    tmp_posi = df[df['일자'] == filter_alert_list.iloc[i,4]].index[0]

    dff= df.iloc[:tmp_posi]
    dff = dff[dff['고가']!=0].reset_index(drop=True)

    real_date = dff['일자'].iloc[-1]


    df = df[df['고가']!=0]

    posi = df[df['일자'] == real_date].index[0]


  else:


    posi = df[df['일자'] == filter_alert_list.iloc[i,4]].index[0]

    if df.iloc[posi,5]==0:

      dff=df.iloc[:posi]
      dff = dff[dff['고가']!=0].reset_index(drop=True)
      real_date = dff['일자'].iloc[-1]

      df = df[df['고가']!=0].reset_index(drop=True)
      posi = df[df['일자'] == real_date].index[0]


    else:
      df = df[df['고가']!=0].reset_index(drop=True)
      posi = df[df['일자'] == filter_alert_list.iloc[i,4]].index[0]



  if posi-15<0 and posi+17>=df.shape[0]:
    df2 = pd.concat([df.iloc[:posi],df.iloc[posi:]])

    for j in range(posi):
      tmp_str = 'D+'+str(posi-j)
      tmp_df2[tmp_str] = df2['종가'].iloc[j]


    for k in range(df.shape[0]-posi):
      tmp_str = 'D-'+str(k)
      tmp_df2[tmp_str] = df2['종가'].iloc[posi+k]


  elif posi-15<0 and posi+17<df.shape[0]:
    df2 = pd.concat([df.iloc[:posi],df.iloc[posi:posi+17]])

    for j in range(posi):
      tmp_str = 'D+'+str(posi-j)
      tmp_df2[tmp_str] = df2['종가'].iloc[j]


    for k in range(15+2):
      tmp_str = 'D-'+str(k)
      tmp_df2[tmp_str] = df2['종가'].iloc[posi+k]


  elif posi-15>=0 and posi+17>=df.shape[0]:
    df2 = pd.concat([df.iloc[posi-15:posi],df.iloc[posi:]])

    for j in range(15):
      tmp_str = 'D+'+str(15-j)
      tmp_df2[tmp_str] = df2['종가'].iloc[j]

    for k in range(df.shape[0]-posi):
      tmp_str = 'D-'+str(k)
      tmp_df2[tmp_str] = df2['종가'].iloc[15+k]

  else:
    df2 = pd.concat([df.iloc[posi-15:posi],df.iloc[posi:posi+17]])



    for j in range(15):
      tmp_str = 'D+'+str(15-j)
      tmp_df2[tmp_str] = df2['종가'].iloc[j]


    for k in range(15+2):
      tmp_str = 'D-'+str(k)
      tmp_df2[tmp_str] = df2['종가'].iloc[15+k]


  tmp_df = pd.concat([tmp_df, tmp_df2],ignore_index=True)



final_result = pd.concat([tmp_df.iloc[:,:5],(tmp_df.iloc[:,5:].diff(axis=1)/tmp_df.iloc[:,5:].shift(1,axis=1)*100).iloc[:,1:]], axis=1)



import dash
from dash import html

app = dash.Dash(__name__)
server = app.server


app.title = "Market Alert Summary"


# bar chart용 데이터셋 만들기
bar_data = alert_list[alert_list['지정일'].isin(list(alert_list['지정일'].unique())[:60])]

df = bar_data.pivot_table(index='시장경보', columns='지정일', values='종목코드', aggfunc='count').fillna(0)

tmp = ['투자주의종목','투자경고종목','투자위험종목']

tmp2 = list(df.index)

df3 = [x for x in tmp if not any(y == x for y in tmp2)]

df2 = pd.DataFrame(index=df3, columns=df.columns).fillna(0)

result = pd.concat([df,df2])

bar_data2=result.loc[tmp]

bar_data2 = bar_data2.stack().reset_index()

bar_data2.columns=['시장경보', '지정일', '지정건수']

fig = px.bar(bar_data2, x='지정일', y='지정건수', color="시장경보",  text="지정건수", color_discrete_sequence=["#03fc56", "#0317fc", "#fc0303"],)

fig.update_layout(
  title=dict(text = ' <b> 최근 2개월 Trend (일별) </b>', x=0.5, font=dict(family='Courier New', size=13, color='black')),
  # paper_bgcolor='#fbf7fc', # 차트 바깥쪽 배경색
  # legend=dict(orientation='v', xanchor='left', x=0.01, yanchor='bottom', y=0.9, font=dict(family='Courier New', size=14, color='black')),
  # height=500, width=750,
  # paper_bgcolor='#171b26', # 차트 바깥쪽 배경색
  # plot_bgcolor='#171b26'
)



# 파이 차트 생성


tmp_dff = pd.DataFrame([[0,0,0]],columns=['투자주의종목','투자경고종목','투자위험종목'])

tmp_dff2 = pd.DataFrame(data=[today_alert_list['시장경보'].value_counts().values], columns=list(today_alert_list['시장경보'].value_counts().index))

tmp_dff3 = pd.concat([tmp_dff,tmp_dff2])

tmp_dff4 = pd.DataFrame(tmp_dff3.sum()).reset_index()

tmp_dff4.columns=['시장경보','지정 종목 수']


# fig2 = go.Figure(data=[go.Pie(labels=list(tmp_dff3.sum().index), values=list(tmp_dff3.sum().values), hole=.4,  )])

fig2 = px.pie(tmp_dff4, values='지정 종목 수', names = '시장경보', color= '시장경보', color_discrete_sequence=["#03fc56", "#0317fc", "#fc0303"] )

fig2.update_traces(hole=.3)

fig2.update_layout(
  title=dict(text = ' <b> 금일 시장경보 지정 종목 수 현황 </b>', x=0.5, font=dict(family='Courier New', size=13, color='black')),
  legend=dict(orientation='v',
              itemsizing = 'trace',
              font=dict(family='Courier New', size=12, color='black',
                                          # xanchor='left', yanchor='top',
                                        ),
            ),
  # height=460, width=460,
  # paper_bgcolor='#fbf7fc', # 차트 바깥쪽 배경색
  # plot_bgcolor='#171b26'
  # colors ={'투자경고종목' : "#0317fc", "투자위험종목":"#fc0303", "투자주의종목" : "#03fc56"}
)



# 월별 시장경보조치 종목수

alert_list['지정연월'] = alert_list['지정일'].str.slice(0, 7)

df22 =alert_list.pivot_table(index='시장경보', columns='지정연월', aggfunc='count').fillna(0)
df22.columns = df22.columns.droplevel()

tmp = ['투자주의종목','투자경고종목','투자위험종목']

tmp2 = list(df22.index)

df3 = [x for x in tmp if not any(y == x for y in tmp2)]




df2 = pd.DataFrame(index=df3, columns=df22.columns).fillna(0)

result = pd.concat([df22,df2])

bar_data3=result.loc[tmp]

bar_data3 = bar_data3.stack().reset_index()

bar_data3.columns=['시장경보', '지정연월', '지정건수']
bar_data3 = bar_data3.sort_values(by=['지정연월'], ascending=True)
bar_data3 = bar_data3.drop_duplicates()
bar_data3 = bar_data3.iloc[3:]


fig6 = px.line(bar_data3, x='지정연월', y='지정건수', color="시장경보",  text="지정건수", color_discrete_map ={'투자경고종목' : "#0317fc", "투자위험종목":"#fc0303", "투자주의종목" : "#03fc56"},)

fig6.update_layout(
  title=dict(text = ' <b> 최근 1년 Trend (월별) </b>', x=0.5, font=dict(family='Courier New', size=13, color='black')),
  # paper_bgcolor='#fbf7fc', # 차트 바깥쪽 배경색
  # legend=dict(orientation='v', xanchor='left', x=0.01, yanchor='bottom', y=0.9, font=dict(family='Courier New', size=14, color='black')),
  # height=500, width=750,
  # paper_bgcolor='#171b26', # 차트 바깥쪽 배경색
  # plot_bgcolor='#171b26'
)




# 시장별 주가변동률
dfff = final_result[['시장경보','시장구분','D-15','D-14','D-13','D-12','D-11','D-10','D-9','D-8','D-7','D-6','D-5','D-4','D-3','D-2','D-1','D-0','D+1','D+2','D+3','D+4','D+5','D+6','D+7','D+8','D+9','D+10','D+11','D+12','D+13','D+14','D+15']]

dfff2 = dfff.dropna().groupby(['시장경보','시장구분']).mean()

dfff3=dfff2.transpose()

dfff4 = pd.DataFrame(dfff2.stack()).reset_index()

dfff4.columns = ['시장경보','시장구분','날짜','종가변동률_평균']


dfff5 = dfff4[dfff4['시장구분']=='KOSDAQ']

dfff6 = dfff4[dfff4['시장구분']=='KOSPI']

# df = px.data.gapminder().query("continent == 'Oceania'")

fig3 = px.line(dfff5, x='날짜', y='종가변동률_평균', color='시장경보',
              #  facet_row='시장구분',
               markers=True,
               color_discrete_map ={'투자경고종목' : "#0317fc", "투자위험종목":"#fc0303", "투자주의종목" : "#03fc56"}
              #  title='시장경보 지정 전/후 일별 종가변동률'
               )

fig3.update_layout(
  title=dict(text = ' <b> (코스닥) 시장경보지정 전/후 일별 주가변동률 </b>', x=0.5, font=dict(family='Courier New', size=13, color='black')),
  # paper_bgcolor='#fbf7fc', # 차트 바깥쪽 배경색
)



fig7 = px.line(dfff6, x='날짜', y='종가변동률_평균', color='시장경보',
              #  facet_row='시장구분',
               markers=True,
               color_discrete_map ={'투자경고종목' : "#0317fc", "투자위험종목":"#fc0303", "투자주의종목" : "#03fc56"}
              #  title='시장경보 지정 전/후 일별 종가변동률'

               )

fig7.update_layout(
  title=dict(text = ' <b> (코스피) 시장경보지정 전/후 일별 주가변동률 </b>', x=0.5, font=dict(family='Courier New', size=13, color='black')),
  # paper_bgcolor='#fbf7fc', # 차트 바깥쪽 배경색
)



#####################################################################################################################################


app.layout = html.Div([

        html.Div(
            style={
                # 'width': '20%',  # TEST2 공간 너비를 20%로 설정
                'height': '10vh',
                'display': 'flex',
                'flex-direction': 'column',
                # 'align-items': 'center',
                'backgroundColor' : '#3b28c9',
                'border-bottom' : '4px solid #a5a4b0',

            },

            children=[
                html.H1('Market Alert Status Board', style={'color': 'white', "font-weight": "bold", "font-size": 43, "margin-left": 20}),
                # html.Hr(),
                # style={
                #     # 'width': '20%',  # TEST2 공간 너비를 20%로 설정
                #     # 'height': '5%',
                #     # 'display': 'flex',
                #     # 'flex-direction': 'column',
                #     # 'align-items': 'center',
                #     'backgroundColor' : 'yellow'
                # },
            ]
        ),

        html.Div(

            html.Div(

                    children=[
                        html.Div(
                            style={
                                'flex-basis': '0',
                                'flex-grow': 1,
                                'min-width': '33.33%',
                                'max-width': '33.33%',
                                # 'width': '33.33%',
                                'border-right' : '1px solid #a5a4b0',
                            },
                            children=[
                                html.H3('Ⅰ '+datetime.datetime.now().strftime("%Y.%m.%d")+' 시장경보 현황', style={'color': 'black', "font-weight": "bold", "font-size": 19 , "margin-left": 10}),
                                html.Br(),
                                dcc.Graph(
                                    id='graph',

                                    # style={'margin-top': 0},
                                    figure=fig2
                                ),

                                html.H3('※ 시장경보종목 List', style={'color': 'black', "font-weight": "bold", "font-size": 16 , "margin-left": 40}),
                                dcc.Dropdown(
                                        id = 'category',
                                        options = [{ 'label':x, 'value':x} for x in ['투자주의종목','투자경고종목','투자위험종목']],
                                        value = '투자경고종목',
                                        style={"font-size": 11, 'margin' : 'auto'
                                              # 'width':'400px', 
                                        },

                                ),
                                html.Br(),
                                dash_table.DataTable(
                                              id='df_list',

                                              columns = [{"name": i, "id": i} for i in today_alert_list.columns],

                                              data=today_alert_list.to_dict('records'),

                                              page_action='none',

                                              style_table={'overflowY': 'auto',
                                                          #   'width': '400px',
                                                          # 'height': '600px',
                                                           'margin' : 'auto'
                                              },

                                              style_cell={ 'textAlign': 'center', 'color' : 'black', "font-size": 11, 'border': '1px solid gray'},

                                              # style_header={
                                              #             'backgroundColor': 'white',
                                              #             'fontWeight': 'bold'
                                              #               },

                                              style_header={
                                                            # 'backgroundColor': 'rgb(30, 30, 30)',
                                                            'backgroundColor': '#e9e1f7',
                                                            'color': 'black'
                                              },

                                              style_data={
                                                            # 'backgroundColor': 'rgb(50, 50, 50)',
                                                            'backgroundColor': '#fdfcff',
                                                            'color': 'black'
                                              },
                                  )

                            ]
                        ),
                        html.Div(
                            style={
                                'flex-basis': '0',
                                'flex-grow': 1,
                                'min-width': '33.33%',
                                'max-width': '33.33%',
                                'border-right' : '1px solid #a5a4b0',
                                # 'padding' : 5
                            },
                            children=[
                                html.H3('Ⅱ '+'시장경보 지정 Trend', style={'color': 'black', "font-weight": "bold", "font-size": 19 , "margin-left": 10}),
                                html.H3('① 일별 시장경보 지정 현황', style={'color': 'black', "font-weight": "bold", "font-size": 15, "margin-left": 20}),

                                # html.Br(),

                                dcc.Graph(
                                    id='graph2',
                                    figure=fig
                                ),

                                html.H3('② 월간 시장경보 지정 현황', style={'color': 'black', "font-weight": "bold", "font-size": 15, "margin-left": 20}),
                                dcc.Graph(
                                    id='graph6',
                                    figure=fig6
                                )

                              ],         # (7)

                        ),

                        html.Div(
                            style={
                                'flex-basis': '0',
                                'flex-grow': 1,
                                'min-width': '33.33%',
                                'max-width': '33.33%',
                            },
                            children=[
                                html.H3('Ⅲ 시장경보 전/후 종가변동률', style={'color': 'black', "font-weight": "bold", "font-size": 19, "margin-left": 10 }),
                                # html.Br(),
                                html.H3('① KOSDAQ 시장', style={'color': 'black', "font-weight": "bold", "font-size": 15, "margin-left": 20}),
                                dcc.Graph(
                                    id='graph3',
                                    figure=fig3
                                ),

                                html.H3('② KOSPI 시장', style={'color': 'black', "font-weight": "bold", "font-size": 15, "margin-left": 20}),
                                dcc.Graph(
                                    id='graph7',
                                    figure=fig7
                                ),

                                # html.Br(),
                                html.H5('* 최근 1년내 경고/위험 지정 종목 中 ', style={"margin-bottom": 0, "margin-left": 10}),
                                html.H5('  지정 후 15일 이상 매매거래 진행 종목 한정 ', style={"margin-top": 0, "margin-left": 23})
                            ]

                      )
                            ],
                    style={
                          'display': 'flex',
                          'flex-direction': 'row',
                          # 'padding' : '5px',
                          'flex' :1,
                          # 'border' : '4px solid #a5a4b0',
                    },
            ),
            style={
                    # 'width': '100%',  # TEST2 공간 너비를 20%로 설정
                    'height': '100%',
                    # 'display': 'flex',
                    # 'flex-direction': 'row',
                    'backgroundColor' : '#fbf7fc',
                    # 'padding' : '5px',
                    # 'flex' :1,
                    # 'border' : '4px solid #a5a4b0',
                  },

        ),



])

@app.callback(Output("df_list", "data"), Input('category','value'))
def temp(category):

  filtered_data = today_alert_list[today_alert_list['시장경보']==category]

  return filtered_data.to_dict('records')

if __name__ == '__main__':
    app.run_server(port=4444)
