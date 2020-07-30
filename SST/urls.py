from django.urls import path
from . import views

urlpatterns = [
    path('', views.enter, name='enter'),
    path('GoodEnglishModel/', views.verbreplacements_goodenglish, name='goodenglishoutput'),
    path('TensorModel/', views.verbreplacements_tensor, name='tensoroutput'),
    path('Bert/', views.verbreplacements_bert, name='bertoutput'),
    path('OnlineLearning/', views.improvegoodenglish, name='onlinelearning'),
    path('LearningStarted/', views.onlinelearning, name='startlearning'),
    ]