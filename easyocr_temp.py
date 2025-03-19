import easyocr

reader = easyocr.Reader(['ko', 'en'], gpu=True)
results = reader.readtext('AI/image/temp.jpg')

for result in results:
    print(result[1])