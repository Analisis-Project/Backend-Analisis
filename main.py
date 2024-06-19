from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from Models.Graph import Graph
import json
from Models.Matrix import fuerza_bruta
from Models.Matrix_PD_version import bottom_up

Af = {
    '000': {'0': 1, '1': 0},
    '100': {'0': 1, '1': 0},
    '010': {'0': 0, '1': 1},
    '110': {'0': 0, '1': 1},
    '001': {'0': 0, '1': 1},
    '101': {'0': 0, '1': 1},
    '011': {'0': 0, '1': 1},
    '111': {'0': 0, '1': 1}
}

Bf = {
    '000': {'0': 1, '1': 0},
    '100': {'0': 1, '1': 0},
    '010': {'0': 1, '1': 0},
    '110': {'0': 1, '1': 0},
    '001': {'0': 1, '1': 0},
    '101': {'0': 0, '1': 1},
    '011': {'0': 1, '1': 0},
    '111': {'0': 0, '1': 1}
}

Cf = {
    '000': {'0': 1, '1': 0},
    '100': {'0': 0, '1': 1},
    '010': {'0': 0, '1': 1},
    '110': {'0': 1, '1': 0},
    '001': {'0': 1, '1': 0},
    '101': {'0': 0, '1': 1},
    '011': {'0': 0, '1': 1},
    '111': {'0': 1, '1': 0}
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.post("/graph/matrix")
async def matrix(request: Request):
    try:
        body = await request.body()
        data = json.loads(body.decode("utf-8"))

        graph = Graph()
        graph.createFromJson(data)

        matrix = graph.matrix()

        matrix_dict = matrix.to_dict()

        return Response(content=json.dumps(matrix_dict), media_type="application/json", status_code=200)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/graph/random-graph")
async def random_graph(request: Request):
    try:
        body = await request.body()
        data = json.loads(body.decode("utf-8"))

        num_nodes = data['num_nodes']
        complete = data['complete']
        conex = data['conex']
        pondered = data['pondered']
        directed = data['directed']

        graph = Graph()
        graph.randomGraph(num_nodes, complete, conex, pondered, directed)

        resp = graph.toJson()

        return Response(content=json.dumps(resp), media_type="application/json", status_code=200)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/graph/excel-export")
async def excel_export(request: Request):
    try:
        body = await request.body()
        data = json.loads(body.decode("utf-8"))

        graph = Graph()
        graph.createFromJson(data)

        matrix = graph.matrix()

        csv_buffer = matrix.to_csv()

        return StreamingResponse(iter([csv_buffer]), media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="adjacency_matrix.csv"'})
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/graph/save-json")
async def save_json(request: Request):
    try:
        body = await request.body()
        data = json.loads(body.decode("utf-8"))

        graph = Graph()
        graph.createFromJson(data)

        resp = graph.toJson()

        return Response(content=json.dumps(resp), media_type="application/json", status_code=200)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/graph/analytics")
async def analytics(request: Request):
    try:
        body = await request.body()
        data = json.loads(body.decode("utf-8"))

        graph = Graph()
        graph.createFromJson(data)

        resp = graph.analytics()

        return Response(content=json.dumps(resp), media_type="application/json", status_code=200)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/graph/brute-force")
async def brute_force(request: Request):
    try:
        data = await request.json()
        key = data.get('key')
        if not key:
            raise HTTPException(status_code=400, detail="Key is required")

        # Asumiendo que tienes los diccionarios Af, Bf, Cf disponibles aquí
        # De lo contrario, necesitarás obtenerlos o definirlos antes de este paso
        pdox, menor, best_combination, best_dict = fuerza_bruta(Af, Bf, Cf, key=key)

        # Convertir el DataFrame a un formato adecuado para JSON
        result_json = best_dict.to_json(orient="records")
        original_json = pdox.to_json(orient="records")
        
        # Crear un diccionario con todos los resultados
        response_data = {
            "original": original_json,
            "menor": menor,
            "best_combination": best_combination,
            "best_dict": result_json,
            "Af": Af,
            "Bf": Bf,
            "Cf": Cf
        }

        # Convertir el resultado a un formato adecuado para JSON si es necesario
        return Response(content=json.dumps(response_data), media_type="application/json", status_code=200)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/graph/bottom-up")
async def bottom_up_async(request: Request):
    try:
        data = await request.json()
        key = data.get('key')
        if not key:
            raise HTTPException(status_code=400, detail="Key is required")

        # Asumiendo que tienes los diccionarios Af, Bf, Cf disponibles aquí
        # De lo contrario, necesitarás obtenerlos o definirlos antes de este paso
        pdox, menor, best_combination, best_dict = bottom_up(Af, Bf, Cf, key=key)

        # Convertir el DataFrame a un formato adecuado para JSON
        result_json = best_dict.to_json(orient="records")
        original_json = pdox.to_json(orient="records")
        
        # Crear un diccionario con todos los resultados
        response_data = {
            "original": original_json,
            "menor": menor,
            "best_combination": best_combination,
            "best_dict": result_json,
            "Af": Af,
            "Bf": Bf,
            "Cf": Cf
        }

        # Convertir el resultado a un formato adecuado para JSON si es necesario
        return Response(content=json.dumps(response_data), media_type="application/json", status_code=200)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))