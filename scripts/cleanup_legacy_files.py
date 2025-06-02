#!/usr/bin/env python3
"""
Script para limpeza dos arquivos legados do projeto.

Este script identifica e organiza arquivos que não fazem mais parte
da arquitetura principal refatorada.
"""

from pathlib import Path
import shutil

def main():
    """Organiza e limpa arquivos legados."""
    
    print("🧹 LIMPEZA DO PROJETO - Arquivos Legados")
    print("=" * 50)
    
    # Diretório do projeto
    project_root = Path(__file__).parent.parent
    
    # 1. ARQUIVOS LEGADOS PARA MOVER/REMOVER
    legacy_files = [
        # Scripts de correção na raiz (mover para scripts/legacy/)
        "run_pipeline.py",
        "fix_preprocessor.py", 
        "fix_preprocessor_simple.py",
        
        # Módulos antigos em src/ (mover para src/legacy/)
        "src/data_loader.py",
        "src/preprocessing.py",
        "src/model_training.py",
        "src/evaluation.py",
        "src/predict.py",
        "src/config.py",
        "src/__init__.py"
    ]
    
    # 2. CRIAR ESTRUTURA DE LEGACY
    legacy_scripts_dir = project_root / "scripts" / "legacy"
    legacy_src_dir = project_root / "src" / "legacy"
    
    print("📂 Criando diretórios para arquivos legados...")
    legacy_scripts_dir.mkdir(parents=True, exist_ok=True)
    legacy_src_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. MOVER ARQUIVOS LEGADOS
    moved_files = []
    
    for file_path in legacy_files:
        source = project_root / file_path
        
        if source.exists():
            if file_path.startswith("src/"):
                # Mover módulos antigos para src/legacy/
                dest = legacy_src_dir / source.name
                dest_dir = "src/legacy/"
            else:
                # Mover scripts para scripts/legacy/
                dest = legacy_scripts_dir / source.name
                dest_dir = "scripts/legacy/"
            
            print(f"   📦 {file_path} → {dest_dir}{source.name}")
            
            try:
                # Mover arquivo
                shutil.move(str(source), str(dest))
                moved_files.append((file_path, str(dest)))
            except Exception as e:
                print(f"   ❌ Erro ao mover {file_path}: {e}")
        else:
            print(f"   ⚠️  Arquivo não encontrado: {file_path}")
    
    # 4. CRIAR ARQUIVO README EXPLICATIVO
    legacy_readme = legacy_src_dir / "README.md"
    with open(legacy_readme, 'w', encoding='utf-8') as f:
        f.write("""# Arquivos Legados

Esta pasta contém os módulos da arquitetura antiga do projeto, antes da refatoração.

## ⚠️ IMPORTANTE
Estes arquivos **NÃO devem ser usados** no projeto atual. Eles foram mantidos apenas para referência histórica.

## Arquitetura Atual
Use os módulos em `src/insurance_prediction/` que seguem as melhores práticas de engenharia de software.

## Arquivos Movidos
""")
        
        for old_path, new_path in moved_files:
            if old_path.startswith("src/"):
                f.write(f"- `{old_path}` → `{new_path}`\n")
    
    # 5. CRIAR README PARA SCRIPTS LEGACY
    scripts_readme = legacy_scripts_dir / "README.md" 
    with open(scripts_readme, 'w', encoding='utf-8') as f:
        f.write("""# Scripts Legados

Scripts de correção e manutenção usados durante a transição da arquitetura.

## Scripts
- `run_pipeline.py` - Pipeline antigo (usar `scripts/train_model.py`)
- `fix_preprocessor.py` - Script de correção do preprocessor
- `fix_preprocessor_simple.py` - Versão simplificada da correção

## ⚠️ Aviso
Estes scripts usam imports dos módulos antigos e podem não funcionar mais.
""")
    
    # 6. RESUMO
    print(f"\n✅ LIMPEZA CONCLUÍDA!")
    print(f"   📦 {len(moved_files)} arquivos movidos para pastas legacy")
    print(f"   📂 Criados: scripts/legacy/ e src/legacy/")
    print(f"   📝 READMEs explicativos criados")
    
    print(f"\n🎯 PROJETO AGORA USA APENAS:")
    print(f"   • src/insurance_prediction/ (arquitetura modular)")
    print(f"   • scripts/train_model.py (script principal)")
    print(f"   • tests/ (testes organizados)")
    
    print(f"\n🗑️  PARA REMOVER COMPLETAMENTE:")
    print(f"   rm -rf scripts/legacy/")
    print(f"   rm -rf src/legacy/")

if __name__ == "__main__":
    main() 