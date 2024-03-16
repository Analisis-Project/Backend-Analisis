from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from GraphApp.Models.Graph import Graph
import json

# Create your views here.
def matrix(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            graph = Graph()
            graph.createFromJson(data)

            matrix = graph.matrix()

            matrix_dict = matrix.to_dict()

            return JsonResponse(matrix_dict)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'Método no permitido'}, status=405)
    

def excel_export(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            graph = Graph()
            graph.createFromJson(data)

            matrix = graph.matrix()

            csv_buffer = matrix.to_csv()

            response = HttpResponse(csv_buffer, content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="adjacency_matrix.csv"'

            return response
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'Método no permitido'}, status=405)

def saveJson(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            graph = Graph()
            graph.createFromJson(data)

            json = graph.toJson()

            response = HttpResponse(json, content_type='application/json')
            response['Content-Disposition'] = 'attachment; filename="datos.json"'

            return response
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'Método no permitido'}, status=405)