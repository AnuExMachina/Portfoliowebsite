from django.shortcuts import render

# Create your views here.
def homepage(request):
    return render(request=request,
                  template_name='main/home.html')
                  

def titanic(request):
    return render(request=request,
                  template_name='main/titanic.html')