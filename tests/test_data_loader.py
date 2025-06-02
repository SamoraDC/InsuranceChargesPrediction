#!/usr/bin/env python3

import sys
sys.path.append('.')

from src.data_loader import load_insurance_data
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)

def test_data_loader():
    """Testa o data_loader."""
    try:
        print("🧪 Testando Data Loader...")
        
        # Carregar dados
        data, report = load_insurance_data()
        
        print(f"\n✅ Dados carregados com sucesso!")
        print(f"Shape: {data.shape}")
        print(f"Colunas: {list(data.columns)}")
        
        print(f"\n📊 Primeiras 3 linhas:")
        print(data.head(3))
        
        print(f"\n📋 Relatório de validação:")
        print(f"Warnings: {len(report.get('warnings', []))}")
        print(f"Errors: {len(report.get('errors', []))}")
        
        if report.get('warnings'):
            print("\n⚠️ Warnings encontrados:")
            for warning in report['warnings']:
                print(f"  - {warning}")
        
        if report.get('errors'):
            print("\n❌ Errors encontrados:")
            for error in report['errors']:
                print(f"  - {error}")
        
        print(f"\n📈 Informações dos dados:")
        print(f"Total de linhas: {data.shape[0]}")
        print(f"Total de colunas: {data.shape[1]}")
        print(f"Valores ausentes: {data.isnull().sum().sum()}")
        print(f"Linhas duplicadas: {data.duplicated().sum()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

if __name__ == "__main__":
    success = test_data_loader()
    if success:
        print("\n🎉 Teste do Data Loader concluído com sucesso!")
    else:
        print("\n💥 Teste do Data Loader falhou!") 