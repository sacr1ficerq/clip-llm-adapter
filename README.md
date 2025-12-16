## Что это за репозиторий?

В этом проекте я собрал пайплайн для image captioning на датасете `Flickr8k`, объединив замороженный CLIP-энкодер и замороженную Qwen3-0.6B через собственный адаптер, обучая только этот адаптер.  

Я реализовал `VisionAdapter` для преобразования выходов CLIP в фиксированное число визуальных токенов в пространстве Qwen.
Структура адаптера:

``` python
    x[:, 1:, :].transpose(1, 2) # drop CLS token
    → Conv1d(kernel_size=3, stride=2, padding=1)
    → AdaptiveAvgPool1d(n_tokens)
    → transpose(1, 2)
    → LayerNorm(vision_dim)
    → Linear(vision_dim → qwen_dim)
    → LayerNorm(qwen_dim)
```

Я написал обертку над датасетом и другие вспомогательный функции для обучения.
Также я реализовал небольшой класс `QwenVisionCaptionTrainer`, который строит входы для Qwen из визуальных и текстовых эмбеддингов, считает лосс только по тексту и поддерживает обучение, валидацию и генерацию капшенов.

Если судить по генерируемым капшенам, получилось довольно неплохо, модель действительно обучилась и генерирует близкие к gt описанию капшены:


```
REF: The dogs are in the snow in front of a fence .
HYP: A brown dog is running in the snow . The other dog is a black dog . The dogs are both dogs . The dogs are both dogs . The dogs are both dogs . The dogs are both dogs

REF: a brown and white dog swimming towards some in the pool
HYP: a black and white dog jumping into a pool

REF: A man and a woman in festive costumes dancing .
HYP: A man in a white shirt and a black shirt is holding a sign that says “I’m a man.” ”

REF: A couple of people sit outdoors at a table with an umbrella and talk .
HYP: A man and a woman sit on a bench

REF: A man is wearing a Sooners red football shirt and helmet .
HYP: A black and white football player in a red jersey
```

Для 0.6B модели не так уж и плохо
