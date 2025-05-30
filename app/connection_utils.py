# app/connection_utils.py
import requests
import time
import logging
from app.config import config

logger = logging.getLogger(__name__)

def check_ollama_connection_with_retry(max_retries=3, delay=1.0, timeout=5):
    """
    Prüft Ollama-Verbindung mit Retry-Mechanismus
    """
    base_url = config.get_ollama_base_url()
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Verbindungsversuch {attempt + 1}/{max_retries} zu {base_url}")
            
            response = requests.get(
                f"{base_url}/api/tags", 
                timeout=timeout,
                headers={'Connection': 'close'}  # Verhindert Connection-Pooling-Issues
            )
            
            if response.status_code == 200:
                logger.info(f"Ollama-Verbindung erfolgreich (Versuch {attempt + 1})")
                return True
                
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Verbindungsfehler (Versuch {attempt + 1}): {e}")
        except requests.exceptions.Timeout as e:
            logger.warning(f"Timeout (Versuch {attempt + 1}): {e}")
        except Exception as e:
            logger.warning(f"Unerwarteter Fehler (Versuch {attempt + 1}): {e}")
        
        if attempt < max_retries - 1:
            logger.info(f"Warte {delay} Sekunden vor nächstem Versuch...")
            time.sleep(delay)
    
    logger.error(f"Ollama-Verbindung nach {max_retries} Versuchen fehlgeschlagen")
    return False

def wait_for_ollama_ready(max_wait_time=30, check_interval=2):
    """
    Wartet bis Ollama bereit ist oder Timeout erreicht wird
    """
    start_time = time.time()
    base_url = config.get_ollama_base_url()
    
    logger.info(f"Warte auf Ollama-Bereitschaft unter {base_url}...")
    
    while time.time() - start_time < max_wait_time:
        if check_ollama_connection_with_retry(max_retries=1, timeout=3):
            logger.info("Ollama ist bereit!")
            return True
        
        logger.debug(f"Ollama noch nicht bereit, warte {check_interval} Sekunden...")
        time.sleep(check_interval)
    
    logger.error(f"Timeout: Ollama nicht bereit nach {max_wait_time} Sekunden")
    return False

def get_ollama_status():
    """
    Gibt detaillierten Ollama-Status zurück
    """
    base_url = config.get_ollama_base_url()
    
    try:
        # Basis-Verbindungstest
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return {
                'connected': True,
                'models_count': len(models),
                'status_message': f"✅ Verbunden - {len(models)} Modelle verfügbar"
            }
    except requests.exceptions.ConnectionError:
        return {
            'connected': False,
            'models_count': 0,
            'status_message': f"❌ Verbindungsfehler zu {base_url}"
        }
    except requests.exceptions.Timeout:
        return {
            'connected': False,
            'models_count': 0,
            'status_message': f"⏱️ Timeout bei Verbindung zu {base_url}"
        }
    except Exception as e:
        return {
            'connected': False,
            'models_count': 0,
            'status_message': f"⚠️ Unbekannter Fehler: {str(e)}"
        }
