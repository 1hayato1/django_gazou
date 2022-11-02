from django.shortcuts import render, get_object_or_404
from .forms import UploadForm
from .models import UploadImage

def index(request):
    params = {
        'title': '丸付けする画像のアップロード',
        'upload_form': UploadForm(),
        'id': None,
    }

    if (request.method == 'POST'):
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            upload_image = form.save()

            params['id'] = upload_image.id

    return render(request, 'mojiapp/index.html', params)

def preview(request, image_id=0):

    upload_image = get_object_or_404(UploadImage, id=image_id)

    params = {
        'title': 'この画像を丸付けします',
        'id': upload_image.id,
        'url': upload_image.image.url
    }

    return render(request, 'mojiapp/preview.html', params)