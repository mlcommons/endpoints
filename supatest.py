# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def psycopg3_cursor(conninfo: str):
    """Context manager for psycopg3 cursor that properly handles connection lifecycle.

    Args:
        conninfo: PostgreSQL connection string (DSN or URI).

    Yields:
        A tuple of (cursor, connection) matching the sqlite3_cursor interface.
    """

    print("import psycopg3")
    # conninfo = "postgresql://postgres:[password]@db.[project-ref].supabase.co:5432/postgres"
    #
    # password = "8+_3!KXa+sL$g2x"
    #
    # postgresql://postgres:[YOUR-PASSWORD]@db.lczeskqdhwkfdgbgttqr.supabase.co:5432/postgres
    #
    ###################################################

    import psycopg

    # password = quote_plus("8+_3!KXa+sL$g2x")
    # password1 = "YyM77YSsFGgdkURA"
    password1 = "FkyYF.6G@cW7vj5 "

    # direct connection
    # conninfo1 = "postgresql://postgres:{password}@db.[project-ref].supabase.co:5432/postgres"
    # conninfo1 = f"postgresql://postgres:{password1}@db.lczeskqdhwkfdgbgttqr.supabase.co:5432/postgres"

    # spooler connection
    # or put this on the command lineexamples/01_LocalBenchmark/run_tinyllm.py
    #
    #   postgresql://postgres.lczeskqdhwkfdgbgttqr:YyM77YSsFGgdkURA@aws-1-us-east-2.pooler.supabase.com:6543/postgres

    # 1st WORKING ver.
    conninfo = f"postgresql://postgres.lczeskqdhwkfdgbgttqr:{password1}@aws-1-us-east-2.pooler.supabase.com:6543/postgres"

    print(f"connecting to supabase ORIG {conninfo}")
    # print(f"connecting to supabase NEW {conninfo1}")

    # ORIGINAL but requires coninfo to be passed in correctly
    conn = psycopg.connect(conninfo, autocommit=False)

    # 1st WORKING rev.
    # conn = psycopg.connect(conninfo1, autocommit=False)
    cursor = conn.cursor()
    print(f" psycopg3_cursor: {cursor}")

    #
    try:
        print("supabase: return cursor, conn from iterator")
        yield cursor, conn
    finally:
        print("Cleaning up and getting out of here ")
        cursor.close()
        conn.close()


print(f"name == {__name__}")
if __name__ == "__main__":
    print("start of main")
    cur = psycopg3_cursor("garbage fake string")
    # print(f"end of main. cur: {cur} con: {con}")
    print(f"end of main. cur: {cur} con: ")
