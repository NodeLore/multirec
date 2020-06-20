from django.shortcuts import render
from django.http import HttpResponse
from utils import recommend
import json

# Create your views here.
def index(requests):
    return render(requests, 'index.html', {'movie': recommend.getFirst()})

def recommendMovie(requests):
    model = requests.GET.get("model")
    recType = requests.GET.get("type")
    key = requests.GET.get("key")
    print(model)
    print(recType)
    print(key)
    if recType == "tag":
        return HttpResponse(json.dumps({'result': recommend.recommendByGraphAttr(recType, key)}), content_type='application/json')
    elif recType == "person":
        return HttpResponse(json.dumps({'result': recommend.recommendByGraphAttr(recType, key)}), content_type='application/json')
    elif recType == "storyline":
        if model == 'ArangoDB':
            return HttpResponse(json.dumps({'result': recommend.recommendByGraphStory(key)}), content_type='application/json')
        elif model == 'JCA':
            return HttpResponse(json.dumps({'result': recommend.recommendByJCA(key)}), content_type='application/json')

def queryPerson(requests):
    key = requests.GET.get("key")
    return HttpResponse(json.dumps({'result': recommend.queryPersonByKey(key)}), content_type='application/json')

def queryMovie(requests):
    movieKey = requests.GET.get('key')
    return render(requests, 'index.html', {'movie': recommend.queryMovieByKey(movieKey)})

def searchMovie(requests):
    key = requests.GET.get('keyword')
    return HttpResponse(json.dumps({'result': recommend.queryByKeywords(key)}), content_type='application/json')