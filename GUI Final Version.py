import pickle
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from xgboost import XGBRegressor
import unicodeit

data = pd.read_excel("New Data.xlsx")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
# Load the trained XGBRegressor model
with open('xgboost_CS.pkl', 'rb') as model_file:
    xgboost_loaded = pickle.load(model_file)

# tkinter GUI
root = tk.Tk()
root.title(f"Prediction of Compressive Strength")

canvas1 = tk.Canvas(root, width=550, height=550)
canvas1.configure(background='#e9ecef')
canvas1.pack()

# label0 = tk.Label(root, text='Developed by Mr. Rupesh Kumar', font=('Times New Roman', 15, 'bold'), bg='#e9ecef')
# canvas1.create_window(20, 20, anchor="w", window=label0)
#
# label_phd = tk.Label(root, text='*K. R. Mangalam University, India, Email: tipu0003@gmail.com',
#                      font=('Futura Md Bt', 12), bg='#e9ecef')
# canvas1.create_window(20, 50, anchor="w", window=label_phd)

label_input = tk.Label(root, text='Input Variables', font=('Times New Roman', 12, 'bold', 'italic', 'underline'),
                       bg='#e9ecef')
canvas1.create_window(20, 70, anchor="w", window=label_input)

# Labels and entry boxes
labels = ['% of ceramic in replacement of sand', 'Fly ash (kg)', 'Sand (kg)', 'Ceramic (kg)', 'Cement (kg)']

entry_boxes = []
for i, label_text in enumerate(labels):
    label = tk.Label(root, text=unicodeit.replace(label_text), font=('Times New Roman', 12, 'italic'), bg='#e9ecef', pady=5)
    canvas1.create_window(20, 100 + i * 30, anchor="w", window=label)

    entry = tk.Entry(root)
    canvas1.create_window(480, 100 + i * 30, window=entry)
    entry_boxes.append(entry)

label_output = tk.Label(root, text='Compressive Strength, (MPa)', font=('Times New Roman', 12,'bold'),
                        bg='#e9ecef')
canvas1.create_window(50, 520, anchor="w", window=label_output)


def values():
    # Validate and get the values from the entry boxes
    input_values = []
    for entry_box in entry_boxes:
        value = entry_box.get().strip()
        if value:
            try:
                input_values.append(float(value))
            except ValueError:
                messagebox.showerror("Error", "Invalid input. Please enter valid numeric values.")
                return
        else:
            messagebox.showerror("Error", "Please fill in all the input fields.")
            return

    # If all input values are valid, proceed with prediction
    input_data = pd.DataFrame([input_values],
                              columns=X.columns)

    # Predict using the loaded XGBRegressor model
    prediction_result = xgboost_loaded.predict(input_data)
    prediction_result = round(prediction_result[0], 2)

    # Display the prediction on the GUI
    label_prediction = tk.Label(root, text=str(prediction_result), font=('Times New Roman', 20, 'bold'), bg='white')
    canvas1.create_window(300, 520, anchor="w", window=label_prediction)


button1 = tk.Button(root, text='Predict', command=values, bg='#4285f4', fg='white', font=('Times New Roman', 12,'bold'),
                    bd=3, relief='ridge')
canvas1.create_window(480, 520, anchor="w", window=button1)

root.mainloop()
