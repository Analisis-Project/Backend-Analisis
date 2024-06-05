from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from Models.Graph import Graph
import json

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