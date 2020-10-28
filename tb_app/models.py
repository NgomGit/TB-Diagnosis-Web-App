from django.db import models
import time
# Create your models here.
class User(models.Model):
        fullname = models.CharField(max_length=350)
        # username = models.CharField(max_length=350)
        email = models.EmailField(max_length=200)
        phone = models.CharField(max_length=30)
        last_login = models.DateTimeField(auto_now_add=True, blank=True)
