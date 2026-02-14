"""
Serviço para carregar e gerenciar habilidades SAEB/SEAMA/BNCC.
Baseado nos comparativos de matrizes fornecidos.
"""
import json
import os
from typing import Dict, List, Optional
from functools import lru_cache


class SkillsMatrixService:
    """Serviço para gerenciar habilidades das matrizes SAEB/SEAMA."""
    
    def __init__(self):
        self.matrix_path = os.path.join(
            os.path.dirname(__file__), 
            "..", 
            "data", 
            "skills_matrix_saeb_seama.json"
        )
        self._matrix: Optional[Dict] = None
    
    @property
    def matrix(self) -> Dict:
        """Carrega a matriz de habilidades (lazy loading)."""
        if self._matrix is None:
            self._matrix = self._load_matrix()
        return self._matrix
    
    def _load_matrix(self) -> Dict:
        """Carrega o arquivo JSON da matriz."""
        try:
            with open(self.matrix_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Arquivo de matriz não encontrado: {self.matrix_path}")
            return {"grades": {}, "proficiency_levels": {}}
        except json.JSONDecodeError as e:
            print(f"❌ Erro ao decodificar JSON: {e}")
            return {"grades": {}, "proficiency_levels": {}}
    
    def get_skills_by_grade(self, grade: str) -> Dict[str, List[Dict]]:
        """
        Retorna todas as habilidades de um ano escolar.
        
        Args:
            grade: Ex: "9_ano", "5_ano", "2_ano"
            
        Returns:
            Dict com eixos como chaves e lista de habilidades como valores
        """
        grades = self.matrix.get("grades", {})
        if grade not in grades:
            return {}
        return grades[grade].get("skills", {})
    
    def get_skills_by_axis(self, grade: str, axis: str) -> List[Dict]:
        """
        Retorna habilidades de um eixo específico.
        
        Args:
            grade: Ex: "9_ano"
            axis: Ex: "NUMEROS", "ALGEBRA", "GEOMETRIA"
            
        Returns:
            Lista de habilidades do eixo
        """
        skills = self.get_skills_by_grade(grade)
        return skills.get(axis, [])
    
    def get_skill_by_id(self, skill_id: str) -> Optional[Dict]:
        """
        Busca uma habilidade pelo ID.
        
        Args:
            skill_id: Ex: "9N1.1", "9A2.3"
            
        Returns:
            Dict com a habilidade ou None
        """
        for grade_data in self.matrix.get("grades", {}).values():
            for axis_skills in grade_data.get("skills", {}).values():
                for skill in axis_skills:
                    if skill.get("id") == skill_id:
                        return skill
        return None
    
    def get_skills_by_proficiency(self, grade: str, proficiency_level: str) -> List[Dict]:
        """
        Retorna habilidades que incluem um nível de proficiência específico.
        
        Args:
            grade: Ex: "9_ano"
            proficiency_level: Ex: "N3", "N5", "N7"
            
        Returns:
            Lista de habilidades que incluem o nível
        """
        result = []
        for axis_skills in self.get_skills_by_grade(grade).values():
            for skill in axis_skills:
                if proficiency_level in skill.get("proficiency_levels", []):
                    result.append(skill)
        return result
    
    def get_skills_by_saeb_descriptor(self, descriptor: str) -> List[Dict]:
        """
        Retorna habilidades associadas a um descritor SAEB.
        
        Args:
            descriptor: Ex: "D10", "D28", "D36"
            
        Returns:
            Lista de habilidades que incluem o descritor
        """
        result = []
        for grade_data in self.matrix.get("grades", {}).values():
            for axis_skills in grade_data.get("skills", {}).values():
                for skill in axis_skills:
                    if descriptor in skill.get("saeb_2001", []):
                        result.append(skill)
        return result
    
    def format_skill_for_prompt(self, skill: Dict) -> str:
        """
        Formata uma habilidade para uso em prompts.
        
        Returns:
            String formatada com ID e descrição
        """
        skill_id = skill.get("id", "")
        description = skill.get("description", "")
        descriptors = ", ".join(skill.get("saeb_2001", []))
        levels = ", ".join(skill.get("proficiency_levels", []))
        
        result = f"{skill_id} - {description}"
        if descriptors:
            result += f" [Descritores SAEB: {descriptors}]"
        if levels:
            result += f" [Níveis: {levels}]"
        return result
    
    def get_all_skill_ids(self, grade: str = None) -> List[str]:
        """
        Retorna todos os IDs de habilidades.
        
        Args:
            grade: Se especificado, filtra por ano escolar
            
        Returns:
            Lista de IDs de habilidades
        """
        result = []
        grades_to_check = (
            {grade: self.matrix.get("grades", {}).get(grade, {})} 
            if grade else 
            self.matrix.get("grades", {})
        )
        
        for grade_data in grades_to_check.values():
            if not isinstance(grade_data, dict):
                continue
            for axis_skills in grade_data.get("skills", {}).values():
                for skill in axis_skills:
                    result.append(skill.get("id", ""))
        return result
    
    def get_proficiency_level_description(self, level: str) -> str:
        """
        Retorna a descrição de um nível de proficiência.
        
        Args:
            level: Ex: "N3", "N7"
            
        Returns:
            Descrição do nível
        """
        levels = self.matrix.get("proficiency_levels", {})
        level_data = levels.get(level, {})
        return level_data.get("description", f"Nível {level}")


# Instância singleton
_skills_service: Optional[SkillsMatrixService] = None


def get_skills_service() -> SkillsMatrixService:
    """Retorna a instância singleton do serviço."""
    global _skills_service
    if _skills_service is None:
        _skills_service = SkillsMatrixService()
    return _skills_service


# Funções de conveniência
def get_skill_description(skill_id: str) -> str:
    """Retorna a descrição de uma habilidade pelo ID."""
    service = get_skills_service()
    skill = service.get_skill_by_id(skill_id)
    if skill:
        return service.format_skill_for_prompt(skill)
    return skill_id


def list_available_skills(grade: str = "9_ano") -> List[str]:
    """Lista todas as habilidades disponíveis para um ano."""
    service = get_skills_service()
    skills = service.get_skills_by_grade(grade)
    result = []
    for axis, axis_skills in skills.items():
        result.append(f"\n📚 {axis}:")
        for skill in axis_skills:
            result.append(f"  - {service.format_skill_for_prompt(skill)}")
    return result
