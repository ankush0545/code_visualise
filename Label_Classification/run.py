import subprocess
import sys
import os

github_links = [
    "https://github.com/Netflix/metaflow.git",
    "https://github.com/tensorflow/tensorflow.git",
    "https://github.com/pytorch/pytorch.git",
    "https://github.com/scikit-learn/scikit-learn.git",
    "https://github.com/google-deepmind/gemma.git",
    "https://github.com/xbtlin/ai-berkshire.git",
    "https://github.com/google/agents-cli.git",
    "https://github.com/roboflow/supervision.git",
    "https://github.com/shuvonsec/claude-bug-bounty.git",
    "https://github.com/Panniantong/Agent-Reach.git",
    # Frontend
    "https://github.com/facebook/react.git",
    "https://github.com/vuejs/vue.git",
    "https://github.com/angular/angular.git",
    "https://github.com/vercel/next.js.git",
    "https://github.com/sveltejs/svelte.git",
    "https://github.com/remix-run/remix.git",
    "https://github.com/vitejs/vite.git",
    "https://github.com/reduxjs/redux.git",
    "https://github.com/tailwindlabs/tailwindcss.git",
    "https://github.com/ant-design/ant-design.git",
    "https://github.com/mui/material-ui.git",
    "https://github.com/nuxt/nuxt.git",
    # Backend
    "https://github.com/tiangolo/fastapi.git",
    "https://github.com/django/django.git",
    "https://github.com/pallets/flask.git",
    "https://github.com/expressjs/express.git",
    "https://github.com/nestjs/nest.git",
    "https://github.com/koajs/koa.git",
    "https://github.com/hapijs/hapi.git",
    "https://github.com/spring-projects/spring-boot.git",
    "https://github.com/gin-gonic/gin.git",
    "https://github.com/labstack/echo.git",
    "https://github.com/gofiber/fiber.git",
    "https://github.com/actix/actix-web.git",
    "https://github.com/encode/starlette.git",
    "https://github.com/symfony/symfony.git",
    # Database / ORM
    "https://github.com/sqlalchemy/sqlalchemy.git",
    "https://github.com/prisma/prisma.git",
    "https://github.com/typeorm/typeorm.git",
    "https://github.com/sequelize/sequelize.git",
    "https://github.com/Automattic/mongoose.git",
    "https://github.com/cockroachdb/cockroach.git",
    "https://github.com/sqlite/sqlite.git",
    "https://github.com/google/leveldb.git",
    "https://github.com/facebook/rocksdb.git",
    "https://github.com/mongodb/mongo.git",
    "https://github.com/influxdata/influxdb.git",
    "https://github.com/postgres/postgres.git",
    # Cache
    "https://github.com/redis/redis.git",
    "https://github.com/memcached/memcached.git",
    "https://github.com/hazelcast/hazelcast.git",
    "https://github.com/ben-manes/caffeine.git",
    "https://github.com/dragonflydb/dragonfly.git",
    # Queue / Workers
    "https://github.com/celery/celery.git",
    "https://github.com/rabbitmq/rabbitmq-server.git",
    "https://github.com/apache/kafka.git",
    "https://github.com/OptimalBits/bull.git",
    "https://github.com/mperham/sidekiq.git",
    "https://github.com/nsqio/nsq.git",
    "https://github.com/contribsys/faktory.git",
    "https://github.com/apache/rocketmq.git",
    "https://github.com/apache/airflow.git",
    "https://github.com/temporalio/temporal.git",
    "https://github.com/rq/rq.git",
    "https://github.com/resque/resque.git",
    "https://github.com/dramatiq/dramatiq.git",
    "https://github.com/spotify/luigi.git",
    # Gateway / Proxy
    "https://github.com/Kong/kong.git",
    "https://github.com/traefik/traefik.git",
    "https://github.com/envoyproxy/envoy.git",
    "https://github.com/caddyserver/caddy.git",
    "https://github.com/nginx/nginx.git",
    "https://github.com/haproxy/haproxy.git",
    "https://github.com/TykTechnologies/tyk.git",
    # AI / ML
    "https://github.com/keras-team/keras.git",
    "https://github.com/huggingface/transformers.git",
    "https://github.com/dmlc/xgboost.git",
    "https://github.com/microsoft/LightGBM.git",
    "https://github.com/onnx/onnx.git",
    "https://github.com/apache/mxnet.git",
    "https://github.com/explosion/spaCy.git",
    # Utilities
    "https://github.com/lodash/lodash.git",
    "https://github.com/psf/requests.git",
    "https://github.com/encode/httpx.git",
    "https://github.com/pallets/click.git",
    "https://github.com/tj/commander.js.git",
    "https://github.com/yargs/yargs.git",
    "https://github.com/chalk/chalk.git",
    "https://github.com/moment/moment.git",
    "https://github.com/iamkun/dayjs.git",
    "https://github.com/axios/axios.git",
    # Infrastructure
    "https://github.com/curl/curl.git",
    "https://github.com/FFmpeg/FFmpeg.git",
    "https://github.com/grpc/grpc.git",
    "https://github.com/protocolbuffers/protobuf.git",
    "https://github.com/facebook/folly.git",
    "https://github.com/cesanta/mongoose.git",
    "https://github.com/lz4/lz4.git",
    "https://github.com/openssl/openssl.git",
    "https://github.com/kubernetes/kubernetes.git",
    "https://github.com/hashicorp/terraform.git",
    "https://github.com/moby/moby.git",
    "https://github.com/prometheus/prometheus.git",
    "https://github.com/grafana/grafana.git",
    "https://github.com/hashicorp/consul.git",
    "https://github.com/hashicorp/vault.git",
    "https://github.com/istio/istio.git",
    # Testing
    "https://github.com/pytest-dev/pytest.git",
    "https://github.com/jestjs/jest.git",
    "https://github.com/mochajs/mocha.git",
    "https://github.com/cypress-io/cypress.git",
    "https://github.com/microsoft/playwright.git",
    "https://github.com/SeleniumHQ/selenium.git",
    "https://github.com/junit-team/junit5.git",
    "https://github.com/vitest-dev/vitest.git",
    # Mobile
    "https://github.com/flutter/flutter.git",
    "https://github.com/facebook/react-native.git",
    "https://github.com/ionic-team/ionic-framework.git",
    "https://github.com/expo/expo.git",
    # Game dev
    "https://github.com/godotengine/godot.git",
    "https://github.com/libgdx/libgdx.git",
    "https://github.com/photonstorm/phaser.git",
    # Security / Crypto
    "https://github.com/OWASP/CheatSheetSeries.git",
    "https://github.com/certbot/certbot.git",
    "https://github.com/letsencrypt/boulder.git",
    "https://github.com/bitcoin/bitcoin.git",
    "https://github.com/ethereum/go-ethereum.git",
    # Data engineering / ETL
    "https://github.com/apache/spark.git",
    "https://github.com/apache/flink.git",
    "https://github.com/dbt-labs/dbt-core.git",
    "https://github.com/getredash/redash.git",
    "https://github.com/apache/superset.git",
    "https://github.com/pandas-dev/pandas.git",
    "https://github.com/numpy/numpy.git",
    "https://github.com/duckdb/duckdb.git",
    # Search
    "https://github.com/elastic/elasticsearch.git",
    "https://github.com/apache/lucene.git",
    "https://github.com/meilisearch/meilisearch.git",
    "https://github.com/typesense/typesense.git",
    # GraphQL
    "https://github.com/graphql/graphql-js.git",
    "https://github.com/apollographql/apollo-server.git",
    "https://github.com/hasura/graphql-engine.git",
    # Serverless / Cloud-native
    "https://github.com/serverless/serverless.git",
    "https://github.com/knative/serving.git",
    "https://github.com/openfaas/faas.git",
    # CI/CD / DevOps
    "https://github.com/ansible/ansible.git",
    "https://github.com/puppetlabs/puppet.git",
    "https://github.com/saltstack/salt.git",
    "https://github.com/argoproj/argo-cd.git",
    "https://github.com/jenkinsci/jenkins.git",
    # Package managers / build tools
    "https://github.com/npm/cli.git",
    "https://github.com/yarnpkg/berry.git",
    "https://github.com/pnpm/pnpm.git",
    "https://github.com/pypa/pip.git",
    "https://github.com/rust-lang/cargo.git",
    "https://github.com/gradle/gradle.git",
    "https://github.com/apache/maven.git",
]


LOG_DIR = "batch_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Persistent cache dir for cloned repos. main.py is expected to forward
# --cache-dir to source_loader.load_sources(), which (as of the patched
# source_loader.py) now clones straight into this dir instead of a
# throwaway temp dir when one is given -- so every repo here gets cloned
# ONCE across this whole run and stays on disk afterward. That matters
# because build_dataset.py needs the actual file content later to build
# the CodeBERT training set, and re-cloning ~150 repos (tensorflow,
# pytorch, kubernetes, postgres, ...) a second time would be slow and
# wasteful. Point this somewhere with enough disk space.
CACHE_DIR = os.path.expanduser("~/repo_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

BATCH_SIZE = 5
batch_results = []  # (batch_index, urls, returncode)

for batch_index, start in enumerate(range(0, len(github_links), BATCH_SIZE), 1):
    batch = github_links[start:start + BATCH_SIZE]

    # List form, no shell=True -- safer, and avoids any shell-quoting
    # surprises with the URLs.
    command = [sys.executable, "main.py"] + batch + ["--cache-dir", CACHE_DIR]
    print(f"\n=== Batch {batch_index}: {len(batch)} repo(s) ===")
    print(" ".join(command))

    # Capture output so a failure doesn't just say "non-zero exit status" --
    # we get to see exactly what main.py printed, and it's also saved to
    # a log file per batch for later inspection.
    log_path = os.path.join(
        LOG_DIR,
        f"batch_{batch_index:02d}.log"
    )

    print(f"Saving logs to: {log_path}")

    with open(log_path, "w", encoding="utf-8") as log_file:

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(line, end="")   # show in terminal
            log_file.write(line)  # save to file

        process.wait()

    result_code = process.returncode

    batch_results.append(
        (batch_index, batch, result_code)
    )

    if result_code != 0:
        print(
            f"✗ Batch {batch_index} failed "
            f"(exit {result_code}) "
            f"-- see {log_path}"
        )
    else:
        print(
            f"✓ Batch {batch_index} completed "
            f"-- see {log_path}"
        )

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for batch_index, batch, rc in batch_results:
    status = "OK" if rc == 0 else f"FAILED (exit {rc})"
    print(f"  Batch {batch_index:2d} [{status:14s}] {len(batch)} repo(s)")

n_ok = sum(1 for _, _, rc in batch_results if rc == 0)
print(f"\n{n_ok}/{len(batch_results)} batches completed successfully.")
print("Finished running main.py with all repositories completed.")