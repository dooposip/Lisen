@echo off
:: ============================================
:: 🌙 LISEN 행동 인식 AI Dashboard 자동 실행기
:: ============================================

:: 현재 bat 파일이 있는 디렉토리 기준으로 이동
cd /d "%~dp0"

:: LISEN 폴더로 이동
cd LISEN

title 🌙 LISEN 행동 인식 AI Dashboard
echo ===============================================
echo   LISEN Streamlit Dashboard 실행 중...
echo ===============================================

:: Conda 환경 활성화
call conda activate test_stream
set CUDA_VISIBLE_DEVICES=0
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv

:: Streamlit 실행
streamlit run "interfaces/streamlit_app/app.py" --server.port 8501

pause
