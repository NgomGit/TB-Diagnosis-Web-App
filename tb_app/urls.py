from django.urls import path, include
from django.conf.urls import url
from . import views

urlpatterns = [
    # ex: /polls/
    url(r"^accounts/", include("django.contrib.auth.urls")),
    url(r"^dashboard/", views.dashboard, name="dashboard"),
    url(r"^register/", views.register, name="register"),
    path('', views.index, name='index'),
   
   
]
