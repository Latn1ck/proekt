from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
key=os.getenv("DEEPSEEK_KEY")
client=OpenAI(api_key=key,base_url="https://api.deepseek.com")
N=10000
reply=client.chat.completions.create(model="deepseek-v4-pro",
    messages=[{'role':'system','content':f'Тебе нужно генерировать по {N} образцов класса на русском языке, который введёт пользователь, учитывая подклассы. Выводить каждый образец с новой строки без нумерации, через пробел код.'},
              {'role':'user','content':'Класс "Руды, шлак и зола", подклассы: 2603 Руды и концентраты медные, 2604 Руды и концентраты никелевые, 2605 Руды и концентраты кобальтовые, 2606 Руды и концентраты алюминиевые'}
              ])
print(reply)