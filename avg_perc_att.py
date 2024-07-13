import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statistics

text_path = os.path.join(os.getcwd(),'experiments/validation/mlm_task/files/txt')
attentions_distribution = []
for i in range(9831):
    mask_text_path = os.path.join(text_path,f"gm_{i}.txt")
    #print(mask_text_path)
    if f"gm_{i}.txt" in os.listdir(text_path):
        with open(mask_text_path,'r') as f:
            perc_att = float(f.readlines()[1].split(":")[-1])
            attentions_distribution.append(perc_att)
mean= sum(attentions_distribution)/len(attentions_distribution)
median = statistics.median(attentions_distribution)
print(f'means is {mean}')
print(f'median is {median}')
attentions_distribution = pd.DataFrame({'attention distribution':attentions_distribution})
sns.kdeplot(data=attentions_distribution, x='attention distribution')
plt.title('Attention Distribution KDE Plot')
plt.xlabel('Attention Distribution')
plt.ylabel('Density')
plt.show()

