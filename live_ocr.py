import cv2
import easyocr
import pandas as pd

reader = easyocr.Reader(['en'])

harcascade = "model/haarcascade_russian_plate_number.xml"

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)
count = 0
min_area = 500

while True:
    success, img = cap.read()

    plate_cascade = cv2.CascadeClassifier(harcascade)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    img_roi = None  # Initialize img_roi here

    for (x, y, w, h) in plates:
        area = w * h

        if area > min_area:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Number plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
            img_roi = img[y:y + h, x:x + w]
            cv2.imshow("ROI", img_roi)

    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('s') and img_roi is not None:
        # Save the plate image
        plate_image_path = "plate_img/scaned_img_" + str(count) + ".jpeg"
        cv2.imwrite(plate_image_path, img_roi)

        # Perform OCR using EasyOCR on the saved image
        result = reader.readtext(plate_image_path)

        extracted_text = ' '.join([res[1] for res in result])
        print("Extracted Text:", extracted_text)  # Debugging output

        # Save the extracted text to a CSV file
        csv_file = 'extracted_plate_text.csv'

        if extracted_text.strip():  # Check if extracted_text is not empty or only whitespace
            with open(csv_file, 'a') as file:
                file.write(extracted_text + '\n')

            # Display 'Plate Saved' message and update count
            cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
            cv2.imshow("Results", img)
            cv2.waitKey(500)
            count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
