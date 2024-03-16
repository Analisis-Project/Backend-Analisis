from django.shortcuts import render
from django.http import JsonResponse
from Models import Graph
import json

# Create your views here.
def save(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            graph = Graph()

            for node in data['nodes']:
                graph.addNode(node['id'], node['value'], node['label'], node['data'], node['type'], node['radius'], node['coordenates'])

            for edge in data['edges']:
                graph.addEdge(edge['type'], edge['source'], edge['target'], edge['weight'], edge['directed'])

            return JsonResponse({'message': 'Grafo guardado correctamente'})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'Método no permitido'}, status=405)