# Сборка галереи для GitHub Pages

Каталог `pages/` — единственное место, где живёт сборка страницы. Сам HTML
здесь не хранится: он каждый раз собирается из `sweep/`, `runs/` и
`web.js`-модулей экспериментов и уходит артефактом в GitHub Actions.

## Локально

```bash
cd try_3_code
pages/build.sh                 # -> pages/_site (этот каталог игнорируется git)
pages/build.sh /tmp/sweep-site # или в любой другой каталог
```

`pages/build.sh` — тонкая обёртка над `python3 -m sweep export --runs runs`.
Нужны зависимости из `requirements.txt` и Python 3.10+; `node` желателен —
сборка проверяет собранный `assets/rules.js` через `node --check`, а без node
просто пропускает эту проверку.

Открыть результат: `pages/_site/index.html`. Если браузеру нужен http-origin
(например, для `fetch`), поднимите статику из каталога сборки.

## На GitHub

Workflow `.github/workflows/pages.yml` в корне репозитория:

1. на push в `main` (и вручную через `workflow_dispatch`) ставит Python и Node;
2. выполняет `try_3_code/pages/build.sh "$RUNNER_TEMP/site"`;
3. публикует каталог через `actions/upload-pages-artifact` + `actions/deploy-pages`.

Включается один раз: Settings → Pages → Source: **GitHub Actions**. Отдельной
ветки `gh-pages` нет, поэтому в истории репозитория не оседают мегабайты
готовой страницы.

## Почему не ветка `gh-pages`

Ветка `gh-pages` хранит каждый вариант страницы в истории git: около 8 МБ на
сборку, и каждый новый прогон перебора добавляет ещё один вариант. Артефакт
Actions существует только у конкретного запуска, а опубликованная версия
заменяется целиком, не оставляя следов в истории.
