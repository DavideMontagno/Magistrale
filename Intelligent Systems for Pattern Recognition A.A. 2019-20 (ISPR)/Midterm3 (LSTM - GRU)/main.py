 
import os

############### clinton_lstm_no_effects
with open("./result_for_presentation/CLINTON-TRUMP-GRU-EFFECTS/Best_model.txt") as f:
    count=1
    for line in f:
      if(count==3): break
      folder = line.strip()
      for element in os.listdir(folder):
                        tmp = os.path.splitext(element) 
                        if "pt" in tmp[1]:
                          file_tmp = folder+"\\"+str(element)
                          print(file_tmp)
                          print(folder)
                          count+=1
                          os.system("python .\generate_only_text.py {} {} -l 70 --divide 70 --cuda".format(file_tmp,folder))
