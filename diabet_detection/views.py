import os
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.views import View
from .models import UploadedImage
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import cv2
import numpy as np
from django.conf import settings
from django.templatetags.static import static
from django.http import HttpResponse

class HomeView(View):
    def get(self, request):
        return render(request, 'home.html')

class TestModelView(View):
    template_name = 'test_model.html'

    def get(self, request):
        return render(request, self.template_name)

    def post(self, request):
        if request.method == 'POST' and request.FILES['image']:
            uploaded_image = request.FILES['image']
            model_choice = request.POST.get('model', 'simple_ann')  # Default to Simple ANN

            # Process uploaded_image using selected model (Simple ANN, CNN, LSTM)
            result, image_data = self.predict_image(uploaded_image, model_choice)

            # Display the result on the page along with the image data
            return render(request, 'result.html', {'result': result, 'image_data': image_data})
        return render(request, self.template_name)

    def serve_uploaded_image(self, uploaded_image):
        # Open and read the uploaded image file
        with uploaded_image.open() as f:
            image_data = f.read()

        # Set the appropriate content type for the image
        content_type = uploaded_image.content_type

        # Return an HttpResponse with the image data and content type
        return HttpResponse(image_data, content_type=content_type)

    def predict_image(self, uploaded_image, model_choice):
        # Load the selected model
        model_path = os.path.join(settings.BASE_DIR, f'{model_choice}_model.keras')
        selected_model = load_model(model_path)

        # Preprocess the uploaded image
        preprocessed_image = self.preprocess_image(uploaded_image)

        # Make predictions
        predictions = selected_model.predict(preprocessed_image)

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions)

        # Map the index to the corresponding class label
        class_labels = {0: 'class_0', 1: 'class_1'}  # Update with your actual class labels
        predicted_class = class_labels.get(predicted_class_index, 'Unknown')

        # Read the uploaded image data
        uploaded_image.seek(0)  # Ensure file pointer is at the beginning
        image_data = uploaded_image.read()

        return predicted_class, image_data

    def preprocess_image(self, uploaded_image):
        # Convert the uploaded image to a format suitable for model input
        img = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
        img = cv2.resize(img, (28, 28))  # Resize to match the input shape of the model
        img = img / 255.0  # Normalize the pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img

class ResultView(View):
    template_name = 'result.html'

    def get(self, request):
        return render(request, self.template_name)


class ModelPerformanceView(View):
    template_name = 'model_performance.html'

    def get(self, request):
        # Render the initial form
        return render(request, self.template_name)

    def post(self, request):
        # Process the selected model name from the form
        model_name = request.POST.get('model_choice', 'simple_ann', )  # Default to 'simple_ann'

        # Redirect to the detail view with the chosen model name
        return HttpResponseRedirect(reverse('model_performance_detail', args=[model_name]))

class ModelSummaryView(View):

    template_name = 'model_summary.html'
    def get(self, request):
        return render(request, self.template_name)

    def post(self, request):
        chosen_model = request.POST.get('model_choice')
        return render(request, 'model_summary_redirect.html', {'chosen_model': chosen_model})

class ModelPerformanceDetailView(View):
    template_name = 'model_performance_detail.html'

    def get(self, request, model_name):
        # Load the chosen model
        model_path = os.path.join(settings.BASE_DIR, f'{model_name}_model.keras')
        chosen_model = load_model(model_path)

        # Perform actions based on the chosen model for model performance
        # For example, you can evaluate the model on a test dataset and retrieve performance metrics

        # Placeholder for model performance results
        performance_results = {
            'accuracy': 0.85,
            'precision': 0.78,
            'recall': 0.92,
            'f1_score': 0.84,
        }

        # Path to the saved graphic image
        # image_path = os.path.join(settings.MEDIA_ROOT, f'{model_name}_model_performance.png')
        image_url = settings.MEDIA_URL + f'{model_name}_model_performance.png'

        return render(request, self.template_name, {
            'chosen_model': model_name,
            'performance_results': performance_results,
            'image_path': image_url,
        })

class ModelSummaryDetailView(View):
    template_name = 'model_summary_detail.html'

    def get(self, request, model_name):
        # Load the chosen model
        model_path = os.path.join(settings.BASE_DIR, f'{model_name}_model.keras')
        chosen_model = load_model(model_path)

        # Display model summary
        model_summary = []
        chosen_model.summary(print_fn=lambda x: model_summary.append(x))

        return render(request, self.template_name, {
            'chosen_model': model_name,
            'model_summary': model_summary,
        })