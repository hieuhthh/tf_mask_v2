mkdir download

# curl -H "Authorization: Bearer ya29.a0AeTM1ieKWqY_xoWKKze2IORs8-J0yz6YsMOGcFt9depqZzryqTC7Crx1yL6c4ot1owSwwpqABNDjCEYxoUorD4duqRNdLpIVuoD8ze1MURzoE9KFuOFBBtqVFQU8CBhYslGpEmq0PfLXuvblvZjz6fo-z28TaCgYKAbQSARASFQHWtWOmga6rrcxayb_NEZm6FT2xdw0163" https://www.googleapis.com/drive/v3/files/1Gd1FDFiJ6RyK4mpUc3SohAdgxOExExmS?alt=media -o download/mask_tinh.zip

# Access_Token="ya29.a0AeTM1idhPaNUYtq9xisdL3083NgLIXjSuq34r2UEHK0oWxcuI1IxVQu9gyDFjTmJu23UDX7dpCKqvUCCGbydkg_vStfXqAmJrhKe0Kw5XlhgxhQLqhCbWY7iGEkOS37vzvIPhS5a277omudm3kxAyyUjWmkxaCgYKAc8SARASFQHWtWOmt-PKxEOYkfdccHqPlhEhyQ0163"
# FILE_ID="1Em2UzFHNmygV8POVg7SeAkN6oW6657mT"
# OUTFILE="final.zip"
# curl -H "Authorization: Bearer $Access_Token" "https://www.googleapis.com/drive/v3/files/$FILE_ID?alt=media" -o "download/$OUTFILE"

Access_Token="ya29.a0AeTM1ifWT1r6kVlV1lRaJm3CURlkmcC6W6dRKqAnKYCJxk6qvahcb5PqEvNDkTNQlVyqINJQBjSNv35-h0CXmSR-I0gJKoB5dGJBkHJXPKUz55vfr5EI_Rid0XngQtUIzudtl0sX18cc0LpjUm7iWhiJY5RpaCgYKAS8SARASFQHWtWOmvRReiDuwUrqVSG4uXnB6GQ0163"
FILE_ID="1UR33CAMEz21QqtDL2QXqSo0HdvO0AoTL"
OUTFILE="masked_ms1m.zip"
curl -H "Authorization: Bearer $Access_Token" "https://www.googleapis.com/drive/v3/files/$FILE_ID?alt=media" -o "download/$OUTFILE"