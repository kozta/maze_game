### Bootstrap

O que é Bootstrapping?
Bootstrapping em aprendizado por reforço refere-se à técnica de atualizar estimativas com base em outras estimativas. Em vez de aguardar o resultado final de uma sequência ou episódio (como no método Monte Carlo), algoritmos que usam bootstrapping atualizam suas estimativas de valor (ou valor-ação) a cada passo com base nas estimativas subsequentes. Ele é um conceito central em muitos algoritmos de aprendizado por reforço, particularmente naqueles baseados em diferenças temporais (TD), como Q-learning e SARSA.

Como Funciona?
No contexto de algoritmos como Q-learning e SARSA, o bootstrapping é realizado atualizando a estimativa do valor Q de um estado-ação com base na estimativa Q do próximo estado-ação. Essa atualização é guiada pela Equação de Bellman, que expressa o valor de um estado como uma combinação do reforço imediato e do valor estimado dos estados futuros.

Por exemplo, no Q-learning, a atualização do valor Q para um par estado-ação (s, a) após tomar uma ação e observar a recompensa e o próximo estado seria algo assim:

$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

Onde:

- $\alpha$ é a taxa de aprendizado
- $\gamma$ é o fator de desconto
- $r$ é a recompensa
- $s'$ é o próximo estado
- $a'$ é a próxima ação
- $\max_{a'} Q(s',a')$ é a estimativa do valor Q do próximo estado-ação

Vantagens do Bootstrapping:
- Eficiência: O bootstrapping permite que o agente aprenda mais rapidamente, pois não precisa esperar o fim do episódio para atualizar seus valores de valor-ação.
- Aprendizado em Tempo Real: Em ambientes dinâmicos ou contínuos, onde não é prático esperar o fim de um episódio, o bootstrapping permite que o agente se adapte continuamente com base nas informações mais recentes.
- Eficácia em Ambientes Estocásticos: O bootstrapping pode ajudar o agente a se ajustar melhor em ambientes onde os resultados não são sempre consistentes.

Desvantagens do Bootstrapping
- Propagação de Erro: Inicialmente, as estimativas Q podem ser imprecisas. O bootstrapping pode propagar e amplificar esses erros, especialmente se as estimativas iniciais estiverem muito distantes dos valores verdadeiros.
- Sensibilidade a Hiperparâmetros: Os métodos de bootstrapping são frequentemente sensíveis à configuração de hiperparâmetros como a taxa de aprendizado $(\alpha)$ e o fator de desconto $(\gamma)$.
- Risco de Instabilidade: Em alguns casos, especialmente com configurações inadequadas de parâmetros, o bootstrapping pode levar a instabilidades no aprendizado.

Conclusão para o Relatório
Ao retirar o bootstrapping os algoritmos não conseguem aprender, pois não conseguem atualizar os valores de valor-ação. Isso é especialmente verdadeiro para o SARSA, que é um algoritmo on-policy. O SARSA usa a política atual para escolher a próxima ação e, portanto, precisa de bootstrapping para atualizar os valores de valor-ação. O Q-learning, por outro lado, é um algoritmo off-policy e, portanto, não precisa de bootstrapping para atualizar os valores de valor-ação. No entanto, o bootstrapping ainda é útil para o Q-learning, pois permite que o agente aprenda mais rapidamente e se adapte a ambientes dinâmicos ou contínuos.	