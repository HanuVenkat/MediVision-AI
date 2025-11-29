# helper to delete oldest records if you want to free space
import sqlite3
from pathlib import Path

db = Path(__file__).resolve().parent / 'medivision.db'
conn = sqlite3.connect(str(db))
c = conn.cursor()
# delete oldest if more than 5
c.execute('SELECT COUNT(*) FROM skin_records')
count = c.fetchone()[0]
if count > 5:
    to_delete = count - 5
    c.execute('SELECT id FROM skin_records ORDER BY id ASC LIMIT ?', (to_delete,))
    ids = [str(r[0]) for r in c.fetchall()]
    if ids:
        q = 'DELETE FROM skin_records WHERE id IN ({})'.format(','.join(ids))
        c.execute(q)
        conn.commit()
        print('Deleted', len(ids), 'records')
conn.close()
