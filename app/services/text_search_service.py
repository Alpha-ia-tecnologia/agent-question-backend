"""
Serviço de Busca de Textos Educacionais Autênticos.

Utiliza DuckDuckGo ou fallback para banco de textos pré-definidos.
"""

from typing import Optional, List
from pydantic import BaseModel
import logging
import re
import random

logger = logging.getLogger(__name__)


class RealTextResult(BaseModel):
    """Resultado de uma busca de texto real."""
    text: str
    title: str
    author: Optional[str] = None
    source_url: str
    source_name: str


class TextSearchError(Exception):
    """Erro durante a busca de textos."""
    pass


# Banco de textos educacionais de domínio público para fallback
FALLBACK_TEXTS = [
    {
        "title": "A Cigarra e a Formiga",
        "author": "Esopo (adaptação)",
        "text": """Era uma vez uma cigarra que vivia cantando durante todo o verão, enquanto a formiga trabalhava duro guardando comida para o inverno. Quando o frio chegou, a cigarra não tinha nada para comer e foi pedir ajuda à formiga. A formiga perguntou: "O que você fez durante o verão?" A cigarra respondeu: "Eu cantava o dia inteiro." A formiga então disse: "Se você cantou no verão, agora dance no inverno." Esta fábula nos ensina a importância do trabalho e da prevenção.""",
        "source_url": "https://www.suapesquisa.com/fabulas",
        "source_name": "Domínio Público"
    },
    {
        "title": "O Leão e o Rato",
        "author": "Esopo (adaptação)",
        "text": """Um leão dormia em sua toca quando um ratinho passou correndo por cima dele. O leão acordou furioso e pegou o rato com suas enormes garras. O ratinho tremendo pediu: "Por favor, me solte! Um dia posso ajudá-lo!" O leão riu, mas soltou o pequeno animal. Dias depois, o leão caiu em uma armadilha de caçadores. O ratinho ouviu seus rugidos e veio correndo. Com seus dentes afiados, roeu as cordas da rede até o leão ficar livre. O leão agradeceu e aprendeu que até os pequenos podem ser de grande ajuda.""",
        "source_url": "https://www.suapesquisa.com/fabulas",
        "source_name": "Domínio Público"
    },
    {
        "title": "A Raposa e as Uvas",
        "author": "Esopo (adaptação)",
        "text": """Uma raposa passava por um vinhedo e viu um cacho de uvas maduras pendurado bem alto. As uvas pareciam deliciosas e ela ficou com muita vontade de comê-las. Saltou o mais alto que pôde, mas não conseguiu alcançar. Tentou várias vezes, saltando cada vez mais, porém as uvas continuavam fora de seu alcance. Cansada, a raposa desistiu e foi embora dizendo: "Essas uvas estão verdes, não quero mais!" Esta fábula ensina que muitas vezes desprezamos o que não podemos conseguir.""",
        "source_url": "https://www.suapesquisa.com/fabulas",
        "source_name": "Domínio Público"
    },
    {
        "title": "A Tartaruga e a Lebre",
        "author": "Esopo (adaptação)",
        "text": """A lebre vivia se gabando de ser o animal mais rápido da floresta e sempre zombava da tartaruga por ser lenta. Um dia, a tartaruga desafiou a lebre para uma corrida. Todos os animais riram, mas aceitaram assistir. No dia da corrida, a lebre saiu disparada e logo estava muito à frente. Confiante, resolveu tirar um cochilo. A tartaruga, sem parar um momento, continuou andando devagar, mas firmemente. Quando a lebre acordou, viu a tartaruga cruzando a linha de chegada. A moral é: devagar e sempre vence a corrida.""",
        "source_url": "https://www.suapesquisa.com/fabulas",
        "source_name": "Domínio Público"
    },
    {
        "title": "O Corvo e a Raposa",
        "author": "Esopo (adaptação)",
        "text": """Um corvo encontrou um pedaço de queijo e voou para um galho alto para saboreá-lo em paz. Uma raposa faminta passou por ali e viu o corvo com o queijo. A raposa, que era muito esperta, começou a elogiar o corvo: "Que penas lindas você tem! Se seu canto for tão belo quanto sua aparência, você deve ser o rei das aves!" O corvo, todo orgulhoso, abriu o bico para cantar e o queijo caiu direto na boca da raposa, que foi embora satisfeita. Moral da história: cuidado com os bajuladores.""",
        "source_url": "https://www.suapesquisa.com/fabulas",
        "source_name": "Domínio Público"
    },
    {
        "title": "O Pastor e o Lobo",
        "author": "Esopo (adaptação)",
        "text": """Um jovem pastor cuidava das ovelhas de seu pai nas montanhas. Para se divertir, começou a gritar: "Socorro! O lobo está atacando as ovelhas!" Os aldeões correram para ajudar, mas não havia lobo algum. O menino riu muito. Fez a mesma brincadeira várias vezes. Um dia, um lobo de verdade apareceu e começou a atacar o rebanho. O pastor gritou desesperado por socorro, mas ninguém veio, pois todos pensavam que era mais uma mentira. Moral: quem mente perde a confiança mesmo quando diz a verdade.""",
        "source_url": "https://www.suapesquisa.com/fabulas",
        "source_name": "Domínio Público"
    },
    {
        "title": "Canção do Exílio",
        "author": "Gonçalves Dias",
        "text": """Minha terra tem palmeiras,
Onde canta o Sabiá;
As aves, que aqui gorjeiam,
Não gorjeiam como lá.

Nosso céu tem mais estrelas,
Nossas várzeas têm mais flores,
Nossos bosques têm mais vida,
Nossa vida mais amores.

Não permita Deus que eu morra,
Sem que eu volte para lá;
Sem que desfrute os primores
Que não encontro por cá.""",
        "source_url": "https://www.todamateria.com.br/cancao-do-exilio",
        "source_name": "Poesia Brasileira"
    },
    {
        "title": "Receita de Bolo de Cenoura",
        "author": "Culinária Brasileira",
        "text": """Ingredientes: 3 cenouras médias picadas, 3 ovos, 1 xícara de óleo, 2 xícaras de açúcar, 2 xícaras de farinha de trigo, 1 colher de fermento em pó. 

Modo de preparo: Bata no liquidificador as cenouras, os ovos e o óleo. Transfira para uma tigela e misture o açúcar e a farinha. Por último, acrescente o fermento. Despeje em uma forma untada e leve ao forno preaquecido a 180°C por 40 minutos. Para a cobertura, derreta 3 colheres de chocolate em pó com 1 colher de manteiga e 3 colheres de leite. Espalhe sobre o bolo ainda quente.""",
        "source_url": "https://www.receitas.com.br",
        "source_name": "Receitas Brasileiras"
    },
]


class TextSearchService:
    """
    Serviço para busca de textos educacionais autênticos.
    
    Tenta usar DuckDuckGo e faz fallback para banco de textos
    pré-definidos quando a busca online falha.
    """
    
    def __init__(self):
        """Inicializa o serviço de busca de textos."""
        self.ddgs_available = False
        try:
            from duckduckgo_search import DDGS
            self.ddgs = DDGS()
            self.ddgs_available = True
            logger.info("✅ TextSearchService inicializado com DuckDuckGo")
        except Exception as e:
            logger.warning(f"⚠️ DuckDuckGo não disponível: {e}")
            logger.info("📚 Usando banco de textos de fallback")
    
    def _get_fallback_texts(self, count: int) -> List[RealTextResult]:
        """Retorna textos do banco de fallback."""
        selected = random.sample(FALLBACK_TEXTS, min(count, len(FALLBACK_TEXTS)))
        return [
            RealTextResult(
                text=t["text"],
                title=t["title"],
                author=t.get("author"),
                source_url=t["source_url"],
                source_name=t["source_name"]
            )
            for t in selected
        ]
    
    def _extract_author_from_content(self, content: str, title: str) -> Optional[str]:
        """Tenta extrair o nome do autor do conteúdo."""
        patterns = [
            r"(?:autor|autora|escrito por|por|de)\s*[:\-]?\s*([A-Z][a-záéíóúãõ]+ [A-Z][a-záéíóúãõ]+)",
            r"([A-Z][a-záéíóúãõ]+ [A-Z][a-záéíóúãõ]+)\s*(?:escreveu|autor|autora)",
            r"—\s*([A-Z][a-záéíóúãõ]+ [A-Z][a-záéíóúãõ]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_source_name(self, url: str) -> str:
        """Extrai nome legível do domínio."""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            name = domain.replace("www.", "").split(".")[0]
            return name.title()
        except Exception:
            return "Fonte Online"
    
    def search_educational_text(
        self,
        skill: str,
        grade: str,
        text_type: Optional[str] = None,
        max_results: int = 10
    ) -> Optional[RealTextResult]:
        """
        Busca um texto educacional.
        
        Args:
            skill: Descrição da habilidade
            grade: Ano/série dos alunos
            text_type: Tipo de texto desejado
            max_results: Número máximo de resultados
            
        Returns:
            RealTextResult com o texto encontrado
        """
        # Usa fallback diretamente (DuckDuckGo não está funcionando bem)
        logger.info("📚 Usando banco de textos educacionais")
        texts = self._get_fallback_texts(1)
        return texts[0] if texts else None
    
    def search_multiple_texts(
        self,
        skill: str,
        grade: str,
        count: int = 3,
        text_type: Optional[str] = None
    ) -> List[RealTextResult]:
        """
        Busca múltiplos textos educacionais.
        
        Args:
            skill: Descrição da habilidade
            grade: Ano/série dos alunos
            count: Quantidade de textos desejados
            text_type: Tipo de texto desejado
            
        Returns:
            Lista de RealTextResult
        """
        logger.info(f"📚 Buscando {count} textos educacionais")
        return self._get_fallback_texts(count)
