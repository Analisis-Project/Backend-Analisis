from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from GraphApp.Models.Graph import Graph
import json

# Create your views here.
@csrf_exempt 
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

@csrf_exempt    
def random_graph(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            num_nodes = data['num_nodes']
            complete = data['complete']
            conex = data['conex']
            pondered = data['pondered']
            directed = data['directed']

            graph = Graph()
            graph.randomGraph(num_nodes, complete, conex, pondered, directed)

            resp = graph.toJson()

            return JsonResponse(resp)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'Método no permitido'}, status=405)
    
@csrf_exempt 
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

@csrf_exempt 
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