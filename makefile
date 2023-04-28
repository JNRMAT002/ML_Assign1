install: venv
	.\venv\Scripts\activate && pip install -r requirements.txt

venv:
	test -d venv || python -m venv venv

clean:
	rd /s /q venv
