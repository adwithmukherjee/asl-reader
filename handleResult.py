from gtts import gTTS
import os

def text2Speach(mytext):
    # This function reads aloud the text that is inputed into it!!!
    
    language = 'en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
    myobj.save("welcome.mp3")
  
    # Playing the converted file
    os.system("mpg321 welcome.mp3")