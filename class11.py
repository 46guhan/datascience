from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Sample text data
text = """Lorem ipsum, dolor sit amet consectetur adipisicing elit. Ipsum est recusandae molestiae pariatur autem reiciendis, temporibus accusantium nulla, assumenda earum aliquid eligendi! Dolorem, eligendi similique praesentium odit ducimus quisquam perferendis.
Enim, sed, asperiores consequatur quae distinctio voluptatem nemo delectus consectetur, suscipit ratione maxime veritatis quam facilis incidunt quod id eligendi rerum! Rerum, quasi accusamus quam fugit quas fuga tempora fugiat.
Possimus, molestiae tempore esse adipisci unde error suscipit iure consectetur vel blanditiis dolore incidunt eaque sapiente. Corporis, eligendi. Illum modi esse assumenda ex, aspernatur porro quisquam ut natus illo dignissimos.
Sapiente sint nulla, dolores eligendi sed laboriosam. Maiores quod iure iste magni cupiditate hic saepe exercitationem illum aliquid omnis error, repudiandae explicabo, consequatur qui ipsam molestias architecto vero voluptatibus odio!
Sed dolore tempora sint iure doloremque, magni odit assumenda explicabo ratione voluptatibus voluptatem nam, consequatur nulla inventore iusto quo necessitatibus itaque velit. Inventore labore beatae totam suscipit maxime veritatis maiores?
Quam eos eveniet ipsam dolores! Nesciunt dolor accusamus ipsam, temporibus maiores cum veritatis enim porro autem. Iure expedita corporis eum corrupti dolorum dicta vero tempore cumque, odit vel soluta libero.
Assumenda impedit, doloribus nostrum praesentium ipsam atque tempore! Sed fugit ducimus neque amet labore quo omnis! Tenetur eligendi, necessitatibus pariatur obcaecati deleniti laboriosam iusto consequatur quisquam quasi doloremque? Incidunt, magni!
Doloremque nihil quis recusandae optio quos reprehenderit cumque vitae ipsa debitis, cum nam sapiente qui, provident unde adipisci reiciendis. Totam aperiam quas non tempora repudiandae ab corrupti quod adipisci esse!
Suscipit ullam ex libero laboriosam provident officiis maiores alias, tempora aliquid doloremque aspernatur, id repellendus eos quaerat molestias illum magnam necessitatibus pariatur corrupti. Nisi alias sed sit laboriosam, odio id!
Quis cumque placeat cum qui ea sit, in nesciunt pariatur hic ratione eos repellat aut? Laudantium rem rerum necessitatibus quis. Nobis doloremque aliquam quam. Repudiandae iure vero possimus nulla ab?"""

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Display the word cloud using matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
