# 概要

Docker Compose を使用して、GPUを利用した本リポジトリの環境を構築することができます。

## 動作確認

次の環境で動作確認済みです。

> Docker 20.10.12  
> Docker Compose 2.9.0

## 使い方

次のコマンドで、`lab`サービスをビルドしてコンテナを起動します。


```
cd ./docker
docker-compose build
docker-compose up -d
```

コンテナの起動は、`docker-compose ps`で確認することができます。
```
NAME                COMMAND                  SERVICE             STATUS              PORTS
docker-lab-1        "jupyter notebook --…"   lab                 running             0.0.0.0:6006->6006/tcp, :::6006->6006/tcp, 0.0.0.0:8888->8888/tcp, :::8888->8888/tcp
```

コンテナが立ち上がると、コンテナ内で`Jupyter Notebook`が起動します。
ブラウザから`http://localhost:8888`にアクセスして、`Jupyter Notebook`を開くことができます。
> URLの`localhost`部分は、ご利用の環境に合わせて変更してください。

また、次のコマンドでコンテナの`lab`サービスにbashで入ることができます。

```
docker-compose exec lab bash
```


