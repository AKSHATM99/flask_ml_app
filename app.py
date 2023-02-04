from flask import *
import os
import numpy as np
import pickle
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


UPLOAD_FOLDER = 'static/files/'
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#Loading & Compiling Model 
model = keras.models.load_model('BikeVsCar_Model(91.5).h5') # 90.50% Accurate
model.compile(loss='binary_crossentropy',
             optimizer = 'RMSprop',
             metrics=['accuracy']
             )

#Loading Linear Regression Model
reg = pickle.load(open('linear_reg_model.pkl', 'rb'))

#Route TO HomePage
@app.route("/")
@app.route("/home")
def home():           
    return render_template("homepage.html")

#Image Model Route
@app.route("/image_class")
def image_class():
    return render_template("image_model.html")

#Linear_Reg Model Route
@app.route("/reg_class")
def reg_class():
    return render_template("linear_reg.html")

#Login Page Route
@app.route("/login", methods=['GET', 'POST'])
def login():
    return render_template('login.html')

#Signup Page Route
@app.route("/signup", methods=['GET', 'POST'])
def signup():
    return render_template('signup.html')

#Upload For Image Classification
@app.route("/upload", methods=['POST'])
def upload():
    file = request.files['myfile']
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    # Transforming Image to array for feeding to model
    img = image.load_img("static/files/{}".format(file.filename),target_size=(100, 100))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    images = np.vstack([x])
    val =(model.predict(images))
    if val<0.5:
        os.remove("static/files/{}".format(file.filename))
        return render_template("image_model.html", lable="Bike")
    else:
        os.remove("static/files/{}".format(file.filename))
        return render_template("image_model.html", lable="Car")  

#Upload For Linear Regression
@app.route("/predict", methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_int = [np.array(int_features)]
    prediction = reg.predict(final_int)
    if int(prediction)>0:
        output = str(int(prediction))+ " ₹"
        return render_template('linear_reg.html', output = output)
    else:
        output = "Not A Good Deal"
        return render_template('linear_reg.html', output = output)

    
    
if __name__ == '__main__':
    app.run(debug=True)
