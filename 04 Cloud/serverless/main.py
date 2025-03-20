import functions_framework
import requests
from google.cloud import firestore
from datetime import datetime
import json

# Inicializar el cliente de Firestore
db = firestore.Client()

def save_to_firestore(data):
    """Guarda los datos en Firestore."""
    collection_ref = db.collection('pokemon')
    doc_ref = collection_ref.document()
    doc_ref.set(data)
    return doc_ref.id

@functions_framework.http
def pokemon(request):
    """
    Cloud Function que realiza una petición HTTP y guarda la respuesta en Firestore.
    
    Args:
        request (flask.Request): Objeto request de Flask
            El request debe contener un parámetro 'id' en el body o query string.
    
    Returns:
        dict: Respuesta con el estado de la operación
    """
    try:
        # Obtener el parámetro 'id'
        request_json = request.get_json(silent=True)
        request_args = request.args
        
        if request_json and 'id' in request_json:
            pokemon_id = request_json['id']
        elif request_args and 'id' in request_args:
            pokemon_id = request_args['id']
        else:
            return ({
                'error': 'Se requiere el parámetro "id"'
            }, 400)
        
        # Construir la URL
        url = f'https://pokeapi.co/api/v2/pokemon/{pokemon_id}'
        
        # Realizar la petición HTTP
        response = requests.get(url)
        response.raise_for_status()  # Lanzar excepción si hay error HTTP
        
        # Preparar los datos para Firestore
        data = {
            'url': url,
            'response': response.json(),  # Guardar la respuesta completa como un diccionario
            'status_code': response.status_code,
            'timestamp': datetime.utcnow()
        }
        
        # Guardar en Firestore
        document_id = save_to_firestore(data)
        
        return ({
            'message': 'Datos guardados exitosamente',
            'document_id': document_id,
            'url': url,
            'status_code': response.status_code
        }, 200)
        
    except requests.exceptions.RequestException as e:
        return ({
            'error': f'Error al realizar la petición HTTP: {str(e)}'
        }, 500)
    except Exception as e:
        return ({
            'error': f'Error interno: {str(e)}'
        }, 500) 